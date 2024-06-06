#include <mlir-c/IR.h>
#include <mlir-c/RegisterEverything.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <iostream>
#include <memory>

#include "TensorNetwork/MLIRGen.h"
#include "TensorNetwork/TensorNetworkDialect.h"
#include "TensorNetwork/TensorNetworkOps.h"
#include "llvm/ADT/ArrayRef.h"
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
#include "mlir/IR/Types.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"

namespace py = pybind11;

class OperationWrapper {
   public:
    OperationWrapper(mlir::Operation *op) : op_(op) {}
    mlir::Operation *get() { return op_; }

   private:
    mlir::Operation *op_;
};

class ModuleManager {
   public:
    ModuleManager() {
        ctx_ = std::make_unique<mlir::MLIRContext>();
        ctx_->getOrLoadDialect<mlir::tensor_network::TensorNetworkDialect>();
        ctx_->getOrLoadDialect<mlir::func::FuncDialect>();

        builder_ = std::make_unique<mlir::OpBuilder>(ctx_.get());
        module_ = mlir::ModuleOp::create(builder_->getUnknownLoc());
        builder_->setInsertionPointToStart(module_.getBody());
    }

    mlir::Operation *createIndexOp(int64_t size, const std::string &name) {
        auto returnType = ::mlir::tensor_network::IndexLabelType::get(builder_->getContext());
        auto sizeAttr = builder_->getI64IntegerAttr(size);
        auto nameAttr = builder_->getStringAttr(name);
        auto location = builder_->getUnknownLoc();
        auto indexOp = builder_->create<mlir::tensor_network::IndexOp>(location, returnType, sizeAttr, nameAttr);
        return indexOp;
    }

    // TODO: Create the function overloading with default name
    mlir::Operation *createIndexOp(int64_t size) {
        std::string name = "index_" + std::to_string(nextIndexId_++);
        return createIndexOp(size, name);
    }

    // Still need to add the verification of the values, etc.
    mlir::Operation *createTensorOp(py::array array, py::args pyIndices) {
        // Get the numpy array shape and data
        auto shape = array.shape();
        auto data = static_cast<double *>(array.mutable_data());  // Ensure data is cast to double*

        // Convert the shape to an llvm::ArrayRef
        llvm::ArrayRef<long int> shapeRef(shape, array.ndim());

        // Create a tensor type from the numpy array shape
        auto tensorType = mlir::RankedTensorType::get(shapeRef, builder_->getF64Type());

        // Create a tensor op
        mlir::Location loc = builder_->getUnknownLoc();

        // Ensure data matches the tensorType's element type
        if (tensorType.getElementType().isF64()) {
            llvm::ArrayRef<double> dataRef(data, array.size());
            auto tensorValue = mlir::DenseElementsAttr::get(tensorType, dataRef);

            // Create a vector of index labels from the list of indices
            std::vector<mlir::Value> indexLabels;
            for (py::handle pyIndex : pyIndices) {
                auto indexWrapper = pyIndex.cast<OperationWrapper>();
                mlir::Operation *indexOp = indexWrapper.get();
                assert(indexOp->getNumResults() == 1 && "IndexOp should produce one result");
                mlir::Value indexValue = indexOp->getResult(0);
                indexLabels.push_back(indexValue);
            }

            // Create the tensor op with the tensor type, tensor value, and index labels
            auto tensorOp = builder_->create<mlir::tensor_network::TensorOp>(loc, tensorType, tensorValue, indexLabels);
            return tensorOp;
        } else {
            // Handle error for unsupported tensor element type
            llvm::errs() << "Unsupported tensor element type: " << tensorType.getElementType() << "\n";
            return nullptr;
        }
    }

    mlir::ModuleOp getModule() {
        return module_;
    }

   private:
    std::unique_ptr<mlir::OpBuilder> builder_;
    mlir::ModuleOp module_;
    std::unique_ptr<mlir::MLIRContext> ctx_;
    int nextIndexId_ = 0;
};

PYBIND11_MODULE(tensor_network_ext, m) {
    m.doc() = "TensorNetwork Dialect bindings";

    py::class_<OperationWrapper>(m, "Operation")
        .def(py::init<mlir::Operation *>())
        .def("get", &OperationWrapper::get)
        .def("dump", [](OperationWrapper &self) {
            self.get()->dump();
        });

    py::class_<ModuleManager>(m, "ModuleManager")
        .def(py::init<>())
        .def("Index", [](ModuleManager &self, int64_t size, const std::string &name) {
            return OperationWrapper(self.createIndexOp(size, name));
        })
        .def("Tensor", [](ModuleManager &self, py::array array, py::args indices) {
            return OperationWrapper(self.createTensorOp(array, indices));
        })
        .def("dump", [](ModuleManager &self) {
            self.getModule().dump();
        });
}
