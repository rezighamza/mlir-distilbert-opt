#ifndef DISTILBERT_FUSION_PATTERNS
#define DISTILBERT_FUSION_PATTERNS

include "mlir/IR/PatternBase.td"
include "DistilBERT/IR/DistilBERTOps.td"

//===----------------------------------------------------------------------===//
// Attention Fusion Patterns (Declarative Rewrite Rules)
//===----------------------------------------------------------------------===//

// We want to find the naive pattern:
//   %score = dbert.matmul %query, %key
//   %probs = dbert.softmax %score
//   %out   = dbert.matmul %probs, %value
// And fuse it into:
//   %out = dbert.mha %query, %key, %value

def FuseAttentionPattern : Pat<
  (MatMulOp (SoftmaxOp (MatMulOp $query, $key)), $value),
  (MHAOp $query, $key, $value)
>;

#endif // DISTILBERT_FUSION_PATTERNS
