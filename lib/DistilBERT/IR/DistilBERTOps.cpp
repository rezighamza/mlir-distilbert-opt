#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Builders.h"
#include "DistilBERT/IR/DistilBERTDialect.h.inc"

using namespace mlir;
using namespace mlir::distilbert;

//===----------------------------------------------------------------------===//
// DistilBERT Operations
//===----------------------------------------------------------------------===//

LogicalResult MHAOp::verify() {
  // Example custom verifier: Ensure that Query, Key, and Value have the same element type.
  auto qType = getQuery().getType().dyn_cast<TensorType>();
  auto kType = getKey().getType().dyn_cast<TensorType>();
  
  if (qType && kType && qType.getElementType() != kType.getElementType()) {
    return emitOpError("Query and Key must have the same element type");
  }
  return success();
}

#define GET_OP_CLASSES
#include "DistilBERT/IR/DistilBERTOps.cpp.inc"
