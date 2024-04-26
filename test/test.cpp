#include "TensorNetwork/MLIRGen.h"
#include "TensorNetwork/TensorNetworkDialect.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Parser/Parser.h"

using namespace tensor_network;
namespace cl = llvm::cl;
static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input tensor network file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

int dumpMLIR() {
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::tensor_network::TensorNetworkDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();

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

    module->dump();
    return 0;
}

int main(int argc, char** argv) {
    // // Initialize MLIR context
    // mlir::MLIRContext context;

    // // Create an MLIR builder
    // mlir::OpBuilder builder(&context);

    // context.getOrLoadDialect<mlir::tensor_network::TensorNetworkDialect>();
    // context.getOrLoadDialect<mlir::func::FuncDialect>();

    // Parse the MLIR operation string
    //       const char *mlirCode = R"(
    //   func.func @tensor_contract(%A: tensor<2x3xf32>, %B: tensor<3x4xf32>) -> tensor<2x4xf32> {
    //     %C = "tensor_network.contract"(%A, %B) : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
    //     return %C : tensor<2x4xf32>
    //   }

    // )";
    //       const char *mlirCode = R"(
    //   func.func @tensor_contract(%A: tensor<2x3xf32>, %B: tensor<3x4xf32>) -> tensor<2x4xf32> {
    //     %C = "tensor_network.contract"(%A, %B) : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
    //     return %C : tensor<2x4xf32>
    //   }

    // )";

    //   auto module = mlir::parseSourceString(mlirCode, &context);
    //   if (!module) {
    //       llvm::errs() << "Failed to parse MLIR module\n";
    //       return 1;
    //   }

    //   // Verify the parsed module
    mlir::registerAsmPrinterCLOptions();
    mlir::registerMLIRContextCLOptions();
    cl::ParseCommandLineOptions(argc, argv, "Tensor Network Compiler\n");

    return dumpMLIR();

    return 0;
}
