#ifndef DISTILBERT_OPS_TD
#define DISTILBERT_OPS_TD

include "DistilBERTDialect.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

//===----------------------------------------------------------------------===//
// DistilBERT Dialect Operations
//===----------------------------------------------------------------------===//

def MHAOp : DistilBERT_Op<"mha", [Pure]> {
  let summary = "Multi-Head Attention operation";
  let description = [{
    A high-level representation of Multi-Head Attention.
    Takes Query, Key, and Value tensors and computes the attention output.
  }];

  let arguments = (ins AnyTensor:$query, AnyTensor:$key, AnyTensor:$value);
  let results = (outs AnyTensor:$output);

  let assemblyFormat = "$query `,` $key `,` $value attr-dict `:` type($query) `,` type($key) `,` type($value) `->` type($output)";
  let hasVerifier = 1;
}

def GeLUOp : DistilBERT_Op<"gelu", [Pure, SameOperandsAndResultType]> {
  let summary = "Gaussian Error Linear Unit (GeLU) activation";
  let description = [{
    Applies the GeLU activation function to the input tensor.
  }];

  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$output);

  let assemblyFormat = "$input attr-dict `:` type($input)";
}

def LayerNormOp : DistilBERT_Op<"layernorm", [Pure, SameOperandsAndResultType]> {
  let summary = "Layer Normalization operation";
  let description = [{
    Applies Layer Normalization over a specified dimension.
  }];

  let arguments = (ins AnyTensor:$input, AnyTensor:$weight, AnyTensor:$bias);
  let results = (outs AnyTensor:$output);

  let assemblyFormat = "$input `,` $weight `,` $bias attr-dict `:` type($input)";
}

def EmbeddingsOp : DistilBERT_Op<"embeddings", [Pure]> {
  let summary = "Token and Position Embeddings";
  let description = [{
    Looks up input IDs and returns the fused token + position embeddings.
  }];
  let arguments = (ins AnyTensor:$input_ids);
  let results = (outs AnyTensor:$output);
  let assemblyFormat = "$input_ids attr-dict `:` type($input_ids) `->` type($output)";
}

def SoftmaxOp : DistilBERT_Op<"softmax", [Pure, SameOperandsAndResultType]> {
  let summary = "Softmax over the last dimension";
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$output);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}

def MatMulOp : DistilBERT_Op<"matmul", [Pure]> {
  let summary = "General Matrix Multiplication";
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$output);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($output)";
}

def QuantizeOp : DistilBERT_Op<"quantize", [Pure]> {
  let summary = "Quantize a float tensor to an int8 tensor";
  let description = [{
    Uses a scale factor to quantize f32 values to i8.
  }];
  let arguments = (ins AnyTensor:$input, F32Attr:$scale);
  let results = (outs AnyTensor:$output);
  let assemblyFormat = "$input `scale` `(` $scale `)` attr-dict `:` type($input) `->` type($output)";
}

def DequantizeOp : DistilBERT_Op<"dequantize", [Pure]> {
  let summary = "Dequantize an int8 tensor back to a float tensor";
  let description = [{
    Uses a scale factor to dequantize i8 values back to f32.
  }];
  let arguments = (ins AnyTensor:$input, F32Attr:$scale);
  let results = (outs AnyTensor:$output);
  let assemblyFormat = "$input `scale` `(` $scale `)` attr-dict `:` type($input) `->` type($output)";
}

def AddOp : DistilBERT_Op<"add", [Pure, SameOperandsAndResultType]> {
  let summary = "Element-wise addition";
  let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
  let results = (outs AnyTensor:$output);
  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($output)";
}

def EmbeddingsOp : DistilBERT_Op<"embeddings", [Pure]> {
  let summary = "Embedding lookup table";
  let arguments = (ins AnyTensor:$input_ids);
  let results = (outs AnyTensor:$output);
  let assemblyFormat = "$input_ids attr-dict `:` type($input_ids) `->` type($output)";
}

def GenericOp : DistilBERT_Op<"generic", []> {
  let summary = "A generic fallback for unmapped PyTorch ops";
  let arguments = (ins Variadic<AnyTensor>:$inputs, StringAttr:$target);
  let results = (outs AnyTensor:$output);
  let assemblyFormat = "$inputs attr-dict `:` type($inputs) `->` type($output)";
}

#endif // DISTILBERT_OPS_TD
