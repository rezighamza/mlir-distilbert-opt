#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "DistilBERT/IR/DistilBERTDialect.h.inc"

namespace mlir {
namespace distilbert {

#define GEN_PASS_DEF_DISTILBERTQUANTIZEINT8
#include "DistilBERT/Transforms/Passes.h.inc"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace distilbert {

#define GEN_PASS_DEF_DISTILBERTQUANTIZEINT8
#include "DistilBERT/Transforms/Passes.h.inc"

namespace {

// Rewrite pattern to wrap MatMul with Quantize and Dequantize
struct QuantizeMatMulPattern : public OpRewritePattern<MatMulOp> {
  using OpRewritePattern<MatMulOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(MatMulOp op, PatternRewriter &rewriter) const override {
    auto lhsType = op.getLhs().getType().dyn_cast<TensorType>();
    auto rhsType = op.getRhs().getType().dyn_cast<TensorType>();
    
    // Only apply if inputs are f32. If they are already i8, skip.
    if (!lhsType || !rhsType || !lhsType.getElementType().isF32() || !rhsType.getElementType().isF32()) {
      return failure();
    }

    Location loc = op.getLoc();
    
    // Create the quantized i8 types based on the f32 shapes
    auto i8Type = rewriter.getI8Type();
    auto qLhsType = RankedTensorType::get(lhsType.getShape(), i8Type);
    auto qRhsType = RankedTensorType::get(rhsType.getShape(), i8Type);
    
    auto outType = op.getOutput().getType().dyn_cast<TensorType>();
    auto qOutType = RankedTensorType::get(outType.getShape(), i8Type);

    // --- REAL QUANTIZATION MATH (Symmetric INT8) ---
    // In PTQ, calibration steps attach observed min/max values to the operations.
    // We check if the user/frontend provided "calib_min" and "calib_max".
    float calib_min = -10.0f; // Fallback defaults if not calibrated
    float calib_max = 10.0f;
    
    if (auto minAttr = op->getAttrOfType<FloatAttr>("calib_min")) {
      calib_min = minAttr.getValueAsDouble();
    }
    if (auto maxAttr = op->getAttrOfType<FloatAttr>("calib_max")) {
      calib_max = maxAttr.getValueAsDouble();
    }

    // Symmetric Quantization Formula:
    // We want the range [-max_abs, max_abs] to map to [-127, 127]
    float max_abs = std::max(std::abs(calib_min), std::abs(calib_max));
    
    // Scale = max_abs / (2^(bits-1) - 1)
    // For INT8, this is max_abs / 127.0
    float computed_scale = max_abs / 127.0f;
    
    // Prevent division by zero
    if (computed_scale == 0.0f) computed_scale = 1e-5f;

    FloatAttr scale = rewriter.getF32FloatAttr(computed_scale);

    // Insert Quantize Ops using the mathematically derived scale
    auto qLhs = rewriter.create<QuantizeOp>(loc, qLhsType, op.getLhs(), scale);
    auto qRhs = rewriter.create<QuantizeOp>(loc, qRhsType, op.getRhs(), scale);

    // Insert the i8 MatMul Op
    auto qMatMul = rewriter.create<MatMulOp>(loc, qOutType, qLhs, qRhs);

    // Insert Dequantize Op to return the result back to f32
    auto dequant = rewriter.create<DequantizeOp>(loc, outType, qMatMul, scale);

    // Replace the original f32 MatMul with the dequantized output
    rewriter.replaceOp(op, dequant.getOutput());

    return success();
  }
};

struct DistilBERTQuantizeINT8Pass 
    : public impl::DistilBERTQuantizeINT8Base<DistilBERTQuantizeINT8Pass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    
    patterns.add<QuantizeMatMulPattern>(&getContext());

    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // end anonymous namespace

std::unique_ptr<Pass> createDistilBERTQuantizeINT8Pass() {
  return std::make_unique<DistilBERTQuantizeINT8Pass>();
}

} // namespace distilbert
} // namespace mlir
