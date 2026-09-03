#ifndef DISTILBERT_DIALECT_TD
#define DISTILBERT_DIALECT_TD

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// DistilBERT Dialect Definition
//===----------------------------------------------------------------------===//

def DistilBERT_Dialect : Dialect {
  let name = "dbert";
  let summary = "A custom dialect for DistilBERT optimization";
  let description = [{
    The `dbert` dialect is designed for high-level representations of 
    DistilBERT operations (like attention, layer norm, GeLU) and optimizing 
    them for edge devices, specifically targeting ARM Cortex-A processors.
  }];
  let cppNamespace = "::mlir::distilbert";
}

// Base class for DistilBERT dialect operations.
class DistilBERT_Op<string mnemonic, list<Trait> traits = []> :
    Op<DistilBERT_Dialect, mnemonic, traits>;

#endif // DISTILBERT_DIALECT_TD
