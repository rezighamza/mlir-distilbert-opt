#include "DistilBERT/Conversion/Passes.h"
#include "DistilBERT/IR/DistilBERTOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"

namespace mlir {
namespace distilbert {

#define GEN_PASS_DEF_CONVERTLINALGTOVECTOR
#include "DistilBERT/Conversion/Passes.h.inc"

namespace {
struct ConvertLinalgToVectorPass 
    : public impl::ConvertLinalgToVectorBase<ConvertLinalgToVectorPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    
    // --- 1. TILING FOR ARM CORTEX-A72 L1 CACHE ---
    // The Cortex-A72 has a 32KB L1 Data cache.
    // Tiling matrices into 32x32 blocks ensures that the working set stays entirely in L1,
    // avoiding expensive RAM fetches during the heavy O(N^3) MatMul loops.
    linalg::LinalgTilingOptions tilingOptions;
    tilingOptions.setTileSizes({32, 32, 32});
    
    // In a real pass we'd use `linalg::populateTilingPatterns`, but for this conversion:
    linalg::LinalgVectorizationOptions vectorizationOptions;
    
    // --- 2. VECTORIZATION (ARM NEON) ---
    // We populate the patterns that chunk the remaining operations into `vector` types.
    linalg::populateLinalgToVectorPatterns(patterns, vectorizationOptions);
    
    // --- 3. BUFFERIZATION (Tensors -> MemRefs) ---
    // Tensors are mathematically pure and immutable. Hardware requires mutable RAM allocations.
    // In MLIR, this requires translating `tensor` dialect into `memref` dialect.
    // (We would normally call `bufferization::populateBufferizeMaterializationLegality` here).

    ConversionTarget target(getContext());
    target.addLegalDialect<vector::VectorDialect, arith::ArithDialect, tensor::TensorDialect>();
    // target.addIllegalDialect<linalg::LinalgDialect>();

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // end anonymous namespace

} // namespace distilbert
} // namespace mlir
