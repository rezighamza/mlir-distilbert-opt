#ifndef DISTILBERT_CONVERSION_PASSES_H
#define DISTILBERT_CONVERSION_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace distilbert {

#define GEN_PASS_DECL
#include "DistilBERT/Conversion/Passes.h.inc"

std::unique_ptr<Pass> createConvertDistilBERTToLinalgPass();
std::unique_ptr<Pass> createConvertLinalgToVectorPass();
std::unique_ptr<Pass> createConvertVectorToLLVMPass();

#define GEN_PASS_REGISTRATION
#include "DistilBERT/Conversion/Passes.h.inc"

} // namespace distilbert
} // namespace mlir

#endif // DISTILBERT_CONVERSION_PASSES_H
