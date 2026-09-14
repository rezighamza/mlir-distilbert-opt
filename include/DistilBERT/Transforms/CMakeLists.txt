set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls)
add_public_tablegen_target(DistilBERTPassesIncGen)

set(LLVM_TARGET_DEFINITIONS FusionPatterns.td)
mlir_tablegen(FusionPatterns.h.inc -gen-rewriters)
add_public_tablegen_target(DistilBERTFusionPatternsIncGen)
