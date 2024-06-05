#include <memory>
#include <string>
#include <system_error>

#include "TensorNetwork/MLIRGen.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

// THIS IS CURRENTLY NOT COMPILED; RIGHT NOW THE BINARY COMES FROM test.cpp

using namespace tensor_network;
namespace cl = llvm::cl;
static cl::opt<std::string> inputFilename(cl::Positional,
                                          cl::desc("<input tensor network file>"),
                                          cl::init("-"),
                                          cl::value_desc("filename"));

int dumpMLIR() {
    mlir::MLIRContext context;
    context.getOrLoadDialect<mlir::tensor_network::TensorNetworkDialect>();

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
    mlir::registerASMPrinterCLOptions();
    mlir::registerMLIRContextCLOptions();
    cl::ParseCommandLineOptions(argc, argv, "Tensor Network Compiler\n");

    return dumpMLIR();
}