add_mlir_dialect_library(DistilBERTConversion
  LowerToLinalg.cpp
  LowerToVector.cpp
  LowerToLLVM.cpp

  ADDITIONAL_HEADER_DIRS
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include/DistilBERT/Conversion

  DEPENDS
  DistilBERTConversionPassesIncGen

  LINK_LIBS PUBLIC
  DistilBERTDialect
  MLIRPass
  MLIRTransformUtils
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRVectorDialect
  MLIRLLVMDialect
  MLIRLinalgToStandard
  MLIRVectorToLLVM
  MLIRArithDialect
  MLIRTensorDialect
  MLIRLinalgTransforms
)
