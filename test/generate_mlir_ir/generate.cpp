#include <iostream>
#include <vector>

#include "TensorNetwork/MLIRGen.h"
#include "TensorNetwork/TensorNetworkDialect.h"
#include "TensorNetwork/TensorNetworkOps.h"
#include "TensorNetwork/TensorNetworkTypes.h"
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
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Parser/Parser.h"

using namespace mlir;

void createDummyTensorNetwork(MLIRContext* ctx) {
    ModuleOp module = ModuleOp::create(UnknownLoc::get(ctx));

    // Create a dummy IndexOp with size 4 and no name
    mlir::OpBuilder builder(ctx);

    // Create new Index Operation

    // Create a new IndexOp with size 4 and no name, and specify the return type
    auto location = builder.getUnknownLoc();
    auto returnType = ::mlir::tensor_network::IndexLabelType::get(ctx);

    //StringRef
    auto indexName = builder.getStringAttr("i");

    auto size = builder.getI64IntegerAttr(4);
    auto indexOp = builder.create<mlir::tensor_network::IndexOp>(location, returnType, size, indexName);

    module.push_back(indexOp);
    module.dump();
}

int main() {
    MLIRContext context;
    // register the dialect
    context.getOrLoadDialect<mlir::tensor_network::TensorNetworkDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();

    createDummyTensorNetwork(&context);
}
