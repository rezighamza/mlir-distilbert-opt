#ifndef DISTILBERT_CONVERSION_PASSES
#define DISTILBERT_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ConvertDistilBERTToLinalg : Pass<"convert-dbert-to-linalg", "mlir::ModuleOp"> {
  let summary = "Lower dbert dialect to linalg dialect";
  let description = [{
    Lowers the `dbert` high level operations down to `linalg` structured ops.
  }];
  let constructor = "mlir::distilbert::createConvertDistilBERTToLinalgPass()";
}

def ConvertLinalgToVector : Pass<"convert-linalg-to-vector", "mlir::ModuleOp"> {
  let summary = "Lower linalg dialect to vector dialect (ARM NEON focused)";
  let constructor = "mlir::distilbert::createConvertLinalgToVectorPass()";
}

def ConvertVectorToLLVM : Pass<"convert-vector-to-llvm", "mlir::ModuleOp"> {
  let summary = "Lower vector/memref to LLVM dialect";
  let constructor = "mlir::distilbert::createConvertVectorToLLVMPass()";
}

#endif // DISTILBERT_CONVERSION_PASSES
