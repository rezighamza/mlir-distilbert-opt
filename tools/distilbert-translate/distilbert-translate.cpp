#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetOptions.h"

using namespace mlir;

static llvm::cl::opt<std::string> inputFilename(llvm::cl::Positional,
                                                llvm::cl::desc("<input MLIR file>"),
                                                llvm::cl::init("-"));

static llvm::cl::opt<std::string> outputFilename("o",
                                                 llvm::cl::desc("Output filename"),
                                                 llvm::cl::value_desc("filename"),
                                                 llvm::cl::init("output.o"));

int main(int argc, char **argv) {
  llvm::InitLLVM y(argc, argv);
  
  // Initialize LLVM targets for ARM/AArch64 (and native for testing)
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();

  llvm::cl::ParseCommandLineOptions(argc, argv, "DistilBERT to ARM Object File Translator\n");

  DialectRegistry registry;
  registerLLVMDialectTranslation(registry);

  MLIRContext context(registry);
  
  llvm::SourceMgr sourceMgr;
  auto module = parseSourceFile<ModuleOp>(inputFilename, sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Failed to parse input MLIR file.\n";
    return 1;
  }

  // 1. Translate MLIR to LLVM IR
  llvm::LLVMContext llvmContext;
  auto llvmModule = translateModuleToLLVMIR(module.get(), llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to translate MLIR to LLVM IR.\n";
    return 1;
  }

  // 2. Setup TargetMachine (targeting generic AArch64 for Raspberry Pi)
  std::string error;
  auto targetTriple = llvm::sys::getDefaultTargetTriple(); // For testing locally
  // In production for Raspberry Pi 4, we'd force: targetTriple = "aarch64-linux-gnu";
  
  auto target = llvm::TargetRegistry::lookupTarget(targetTriple, error);
  if (!target) {
    llvm::errs() << error;
    return 1;
  }

  llvm::TargetOptions opt;
  auto RM = llvm::Optional<llvm::Reloc::Model>();
  auto targetMachine = target->createTargetMachine(targetTriple, "generic", "", opt, RM);
  
  llvmModule->setDataLayout(targetMachine->createDataLayout());
  llvmModule->setTargetTriple(targetTriple);

  // 3. Emit Object File
  std::error_code EC;
  llvm::raw_fd_ostream dest(outputFilename, EC, llvm::sys::fs::OF_None);
  if (EC) {
    llvm::errs() << "Could not open file: " << EC.message();
    return 1;
  }

  llvm::legacy::PassManager pass;
  auto FileType = llvm::CGFT_ObjectFile;
  
  if (targetMachine->addPassesToEmitFile(pass, dest, nullptr, FileType)) {
    llvm::errs() << "TargetMachine can't emit a file of this type";
    return 1;
  }

  pass.run(*llvmModule);
  dest.flush();
  
  llvm::outs() << "Successfully emitted object file: " << outputFilename << "\n";
  return 0;
}
