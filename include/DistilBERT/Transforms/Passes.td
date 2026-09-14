#ifndef DISTILBERT_PASSES
#define DISTILBERT_PASSES

include "mlir/Pass/PassBase.td"

def DistilBERTFuseAttention : Pass<"dbert-fuse-attention", "mlir::func::FuncOp"> {
  let summary = "Fuse multiple operations into a single dbert.mha operation";
  let description = [{
    This pass searches for patterns of matrix multiplications, scalings, and 
    softmax operations that correspond to the DistilBERT Multi-Head Attention 
    mechanism and fuses them into a single `dbert.mha` high-level operation 
    for easier target-specific lowering.
  }];
  let constructor = "mlir::distilbert::createDistilBERTFuseAttentionPass()";
}

def DistilBERTQuantizeINT8 : Pass<"dbert-quantize-int8", "mlir::func::FuncOp"> {
  let summary = "Post-training static INT8 quantization";
  let description = [{
    Converts f32 weights and activations in the network to i8 representations
    to save memory bandwidth and compute cycles on the Raspberry Pi.
  }];
  let constructor = "mlir::distilbert::createDistilBERTQuantizeINT8Pass()";
}

#endif // DISTILBERT_PASSES
