#include "DistilBERT/Conversion/Passes.h"
#include "DistilBERT/IR/DistilBERTDialect.h.inc"
#define GET_OP_CLASSES
#include "DistilBERT/IR/DistilBERTOps.h.inc"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace distilbert {

#define GEN_PASS_DEF_CONVERTDISTILBERTTOLINALG
#include "DistilBERT/Conversion/Passes.h.inc"

namespace {

struct MatMulOpLowering : public OpConversionPattern<MatMulOp> {
  using OpConversionPattern<MatMulOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(MatMulOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = op.getOutput().getType().dyn_cast<RankedTensorType>();
    if (!resultType)
      return failure();

    // In a real scenario we'd create linalg.empty to hold the result
    auto emptyTensor = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), resultType.getElementType());
    
    // Fill the empty tensor with 0
    auto zeroAttr = rewriter.getZeroAttr(resultType.getElementType());
    auto zero = rewriter.create<arith::ConstantOp>(loc, zeroAttr);
    auto fillOp = rewriter.create<linalg::FillOp>(loc, zero, emptyTensor);

    // Create linalg.matmul
    auto matmulOp = rewriter.create<linalg::MatmulOp>(
        loc,
        ValueRange{adaptor.getLhs(), adaptor.getRhs()}, // inputs
        ValueRange{fillOp.getResult(0)}                // outputs
    );

    rewriter.replaceOp(op, matmulOp.getResult(0));
    return success();
  }
};

struct ConvertDistilBERTToLinalgPass 
    : public impl::ConvertDistilBERTToLinalgBase<ConvertDistilBERTToLinalgPass> {
  void runOnOperation() override {
    ConversionTarget target(getContext());
    target.addLegalDialect<linalg::LinalgDialect, tensor::TensorDialect, arith::ArithDialect>();
    target.addIllegalDialect<DistilBERT_Dialect>();
    
    RewritePatternSet patterns(&getContext());
    patterns.add<MatMulOpLowering>(patterns.getContext());
    
    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // end anonymous namespace

std::unique_ptr<Pass> createConvertDistilBERTToLinalgPass() {
  return std::make_unique<ConvertDistilBERTToLinalgPass>();
}

} // namespace distilbert
} // namespace mlir
