set(LLVM_LINK_COMPONENTS
  Core
  Support
  Target
  AArch64CodeGen
  AArch64AsmParser
  X86CodeGen
  X86AsmParser
  MC
)

add_mlir_tool(distilbert-translate
  distilbert-translate.cpp
)

target_link_libraries(distilbert-translate PRIVATE
  MLIRIR
  MLIRParser
  MLIRSupport
  MLIRTargetLLVMIRExport
)
