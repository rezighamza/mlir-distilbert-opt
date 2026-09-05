add_mlir_dialect_library(DistilBERTDialect
  DistilBERTDialect.cpp
  DistilBERTOps.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include/DistilBERT/IR

  DEPENDS
  DistilBERTDialectIncGen
  DistilBERTOpsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRInferTypeOpInterface
  MLIRSideEffectInterfaces
)
