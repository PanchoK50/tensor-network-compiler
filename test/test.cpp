#include <string>

#include "TensorNetwork/Passes.h"
#include "TensorNetwork/TensorNetworkDialect.h"
#include "TensorNetwork/TensorNetworkTypes.h"
#include "TensorNetwork/TensorNetworkOps.h"
#include "TensorNetwork/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/TensorToLinalg/TensorToLinalgPass.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/Func/Transforms/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/Passes.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Target/LLVMIR/ModuleImport.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"


using namespace mlir::tensor_network;
namespace cl = llvm::cl;
static cl::opt<std::string>
inputFilename(cl::Positional, cl::desc("<input tensor network file>"),
              cl::init("-"), cl::value_desc("filename"));
static cl::opt<bool> applyLowerings("apply-lowerings",
                                    cl::desc("Apply the lowerings"),
                                    cl::init(false));

void debugPrintModule(const std::string &stepName, mlir::ModuleOp &module) {
    llvm::errs() << "\n=== After " << stepName << " ===\n";
    module->dump();
    llvm::errs() << "\n";
}

int lowerModule(mlir::ModuleOp module, mlir::MLIRContext &context) {
    if (applyLowerings) {
        mlir::PassManager pm(&context);

        pm.addPass(mlir::tensor_network::createTensorNetworkNaiveLoweringPass());
        pm.addPass(mlir::createConvertTensorToLinalgPass());

        // Bufferization
        mlir::bufferization::OneShotBufferizationOptions bufferizationOptions;
        bufferizationOptions.bufferizeFunctionBoundaries = true;
        pm.addPass(mlir::bufferization::createOneShotBufferizePass(bufferizationOptions));

        // Convert Linalg to loops
        pm.addPass(mlir::createConvertLinalgToLoopsPass());

        // Your existing passes
        pm.addPass(mlir::createLowerAffinePass());
        pm.addPass(mlir::createConvertSCFToCFPass());
        pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
        pm.addPass(mlir::createConvertControlFlowToLLVMPass());
        pm.addPass(mlir::createConvertFuncToLLVMPass());
        pm.addPass(mlir::createArithToLLVMConversionPass());
        pm.addPass(mlir::createConvertMathToLLVMPass());
        pm.addPass(mlir::createConvertIndexToLLVMPass());
        pm.addPass(mlir::createConvertVectorToLLVMPass());
        pm.addPass(mlir::createReconcileUnrealizedCastsPass());

        if (mlir::failed(pm.run(module))) {
            llvm::errs() << "Error applying lowerings\n";
            return -1;
        }

        module->dump();
    }
    return 0;
}

int getLLVMIR(mlir::ModuleOp module) {
    /*
   Inspired/Copied from the toy dialect
   */

    llvm::errs() << "getLLVMIR\n";
    module->dump();

    // verify the module first
    if (mlir::failed(mlir::verify(module))) {
        llvm::errs() << "Error in module verification\n";
        return -1;
    }

    llvm::LLVMContext llvmContext;
    auto llvmModule = mlir::translateModuleToLLVMIR(module, llvmContext);
    if (!llvmModule) {
        llvm::errs() << "Failed to emit LLVM IR\n";
        return -1;
    }

    llvm::errs() << *llvmModule << "\n";
    return 0;
}

int main(int argc, char **argv) {
    mlir::registerAsmPrinterCLOptions();
    mlir::registerMLIRContextCLOptions();
    mlir::registerPassManagerCLOptions();
    cl::ParseCommandLineOptions(argc, argv, "Tensor Network Compiler\n");

    mlir::DialectRegistry registry;
    mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
        registry);
    // mlir::func::registerFuncDialectTranslation(registry);
    mlir::registerBuiltinDialectTranslation(registry);
    mlir::registerLLVMDialectTranslation(registry);

    mlir::MLIRContext context(registry);
    context.getOrLoadDialect<mlir::tensor_network::TensorNetworkDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();
    context.getOrLoadDialect<mlir::arith::ArithDialect>();
    context.getOrLoadDialect<mlir::linalg::LinalgDialect>();
    context.getOrLoadDialect<mlir::bufferization::BufferizationDialect>();
    context.getOrLoadDialect<mlir::memref::MemRefDialect>();
    context.getOrLoadDialect<mlir::scf::SCFDialect>();

    // For testing purposes, only accept .mlir code (From Toy Tutorial)
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
        llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
    if (std::error_code ec = fileOrErr.getError()) {
        llvm::errs() << "Could not open input file: " << ec.message() << "\n";
        return -1;
    }
    llvm::SourceMgr sourceMgr;
    sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
    mlir::OwningOpRef<mlir::ModuleOp> module =
        mlir::parseSourceFile<mlir::ModuleOp>(sourceMgr, &context);
    if (!module) {
        llvm::errs() << "Error can't load file " << inputFilename << "\n";
        return 3;
    }

    // Lower to llvm dialect
    if (lowerModule(*module, context) != 0) {
        llvm::errs() << "Error lowering module\n";
        return -1;
    }

    llvm::errs() << "Lowered module\n";
    module->dump();

    // Transform to executable
    if (getLLVMIR(*module) != 0) {
        llvm::errs() << "Error getting LLVM IR\n";
        return -1;
    }

    return 0;
}
