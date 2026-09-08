set(LIBS
  DistilBERTDialect
  DistilBERTTransforms
  MLIROptLib
  MLIRIR
  MLIRSupport
  MLIRPass
  MLIRFuncDialect
  MLIRLinalgDialect
)

add_llvm_executable(distilbert-opt distilbert-opt.cpp)
llvm_update_compile_flags(distilbert-opt)
target_link_libraries(distilbert-opt PRIVATE ${LIBS})
