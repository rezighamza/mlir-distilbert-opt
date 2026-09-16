#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "DistilBERT/IR/DistilBERTDialect.h.inc"
// Define GET_OP_CLASSES here so we can use MHAOp etc if needed.
#define GET_OP_CLASSES
#include "DistilBERT/IR/DistilBERTOps.h.inc"

#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace distilbert {

#define GEN_PASS_DEF_DISTILBERTFUSEATTENTION
#include "DistilBERT/Transforms/Passes.h.inc"

// Include the generated DRR patterns
namespace {
#include "DistilBERT/Transforms/FusionPatterns.h.inc"
}

namespace {
struct DistilBERTFuseAttentionPass 
    : public impl::DistilBERTFuseAttentionBase<DistilBERTFuseAttentionPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    
    // Populate the patterns generated from TableGen (FusionPatterns.td)
    populateWithGenerated(patterns);

    // Apply the patterns greedily to the current function
    if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // end anonymous namespace

std::unique_ptr<Pass> createDistilBERTFuseAttentionPass() {
  return std::make_unique<DistilBERTFuseAttentionPass>();
}

} // namespace distilbert
} // namespace mlir
