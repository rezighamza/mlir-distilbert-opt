#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "DistilBERT/IR/DistilBERTDialect.h.inc"

namespace mlir {
namespace distilbert {
// Forward declarations from Passes.h.inc
#define GEN_PASS_DECL
#include "DistilBERT/Transforms/Passes.h.inc"

#define GEN_PASS_DECL
#include "DistilBERT/Conversion/Passes.h.inc"

void registerDistilBERTPasses() {
#define GEN_PASS_REGISTRATION
#include "DistilBERT/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "DistilBERT/Conversion/Passes.h.inc"
}
} // namespace distilbert
} // namespace mlir

int main(int argc, char **argv) {
  mlir::registerAllPasses();
  mlir::distilbert::registerDistilBERTPasses();

  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  
  // Register our custom dialect
  registry.insert<mlir::distilbert::DistilBERTDialect>();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "DistilBERT optimizer driver\n", registry));
}
