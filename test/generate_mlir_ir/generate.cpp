#include <vector>

#include "TensorNetwork/MLIRGen.h"
#include "TensorNetwork/TensorNetworkDialect.h"
#include "TensorNetwork/TensorNetworkOps.h"
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

#include "mlir/IR/Types.h"
#include "mlir/IR/BuiltinTypes.h"

#include <iostream>

using namespace mlir;

void createDummyTensorNetwork(MLIRContext* ctx) {
    ModuleOp module = ModuleOp::create(UnknownLoc::get(ctx));
    // Create 10 tensors
    std::vector<mlir::tensor_network::TensorOp> tensors;
    OpBuilder builder(ctx);
    for (int i = 0; i < 10; i++) {
        Location loc = UnknownLoc::get(ctx);
        ValueRange operands;
        mlir::Type type = builder.getF64Type();
        ShapedType shapedType = RankedTensorType::get({2, 2}, type);
        llvm::ArrayRef<double> values = {1.0, 2.0, 3.0, 4.0};
        auto valueAttr = DenseElementsAttr::get(shapedType, values);
        // Create the tensor
        auto tensor = builder.create<::mlir::tensor_network::TensorOp>(loc, shapedType, valueAttr);
        tensors.push_back(tensor);
        module.push_back(tensor);
    }

    // Print all the tensors right now:
    for (auto tensor : tensors) {
        std::cout << "Tensor: " << tensor << std::endl;
    }

    for (size_t i = 0; i < tensors.size() - 1; i++) {
        auto lhs = tensors[i];
        auto rhs = tensors[i + 1];
        std::cout << "Contraction between " << lhs << " and " << rhs << std::endl;
        llvm::ArrayRef<Value> contractedIndices = {0, 1};  // contract on the first dimension
        // auto edge = builder.create<::mlir::tensor_network::ContractionEdgeOp>(UnknownLoc::get(ctx), lhs.getResult(), rhs.getResult(), contractedIndices);
        //Print all values for build in order to check for bugs :(
        // std::cout << "lhs: " << lhs.getResult() << std::endl;
        // std::cout << "rhs: " << rhs.getResult() << std::endl;

        // auto lhs_value = lhs.getResult().getDefiningOp()->getAttr("value").cast<DenseElementsAttr>();
        // auto rhs_value = rhs.getResult().getDefiningOp()->getAttr("value").cast<DenseElementsAttr>();

        Location loc = UnknownLoc::get(ctx);

        // The result type should be the result of the contraction
        auto resultType = RankedTensorType::get({2, 2}, builder.getF64Type()); //TODO Correct the shape

        auto edge = builder.create<::mlir::tensor_network::ContractionEdgeOp>(loc, resultType, lhs.getResult(), rhs.getResult());
        module.push_back(edge);
    }

    

    module.dump();
}

int main() {
    MLIRContext context;
    // register the dialect
    context.getOrLoadDialect<mlir::tensor_network::TensorNetworkDialect>();
    context.getOrLoadDialect<mlir::func::FuncDialect>();

    createDummyTensorNetwork(&context);
}