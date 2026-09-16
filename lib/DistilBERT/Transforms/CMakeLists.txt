add_mlir_dialect_library(DistilBERTTransforms
  AttentionFusion.cpp
  QuantizeINT8.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include/DistilBERT/Transforms

  DEPENDS
  DistilBERTPassesIncGen
  DistilBERTFusionPatternsIncGen

  LINK_LIBS PUBLIC
  DistilBERTDialect
  MLIRPass
  MLIRTransformUtils
  MLIRFuncDialect
  MLIRLinalgDialect
)
