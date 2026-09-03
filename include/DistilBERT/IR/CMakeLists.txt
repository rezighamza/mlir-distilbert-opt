set(LLVM_TARGET_DEFINITIONS DistilBERTDialect.td)
mlir_tablegen(DistilBERTDialect.h.inc -gen-dialect-decls)
mlir_tablegen(DistilBERTDialect.cpp.inc -gen-dialect-defs)
add_public_tablegen_target(DistilBERTDialectIncGen)

set(LLVM_TARGET_DEFINITIONS DistilBERTOps.td)
mlir_tablegen(DistilBERTOps.h.inc -gen-op-decls)
mlir_tablegen(DistilBERTOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(DistilBERTOpsIncGen)
