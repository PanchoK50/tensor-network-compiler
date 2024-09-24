#include <mlir-c/IR.h>
#include <mlir-c/RegisterEverything.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Dialect/Func/Extensions/AllExtensions.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/AsyncToLLVM/AsyncToLLVM.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Dialect/Async/Passes.h"
#include "mlir/Dialect/Async/Transforms.h"
#include "mlir/Dialect/ControlFlow/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Target/LLVMIR/Dialect/All.h"
#include "TensorNetwork/Passes.h"
#include "TensorNetwork/TensorNetworkDialect.h"
#include "TensorNetwork/TensorNetworkOps.h"
#include "TensorNetwork/TensorNetworkTypes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/TensorToLinalg/TensorToLinalgPass.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Async/Passes.h"
#include "mlir/Dialect/Async/Transforms.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/Passes.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/ExecutionEngine/CRunnerUtils.h"
#include "mlir/ExecutionEngine/ExecutionEngine.h"
#include "mlir/ExecutionEngine/OptUtils.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/IR/Verifier.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMIRToLLVMTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/LocationSnapshot.h"
#include "mlir/ExecutionEngine/MemRefUtils.h"

namespace py = pybind11;

class IndexLabelTypeWrapper {
public:
    IndexLabelTypeWrapper(mlir::Type indexLabelType)
    : indexLabelType(indexLabelType) {}

    int64_t getSize() const {
        if (auto indexLabel =
            indexLabelType.dyn_cast<mlir::tensor_network::IndexLabelType>()) {
            return indexLabel.getSize().getInt();
        }
        return -1;  // Or throw an exception
    }

    std::string getName() const {
        if (auto indexLabel =
            indexLabelType.dyn_cast<mlir::tensor_network::IndexLabelType>()) {
            return indexLabel.getName().str();
        }
        return "";  // Or throw an exception
    }

    mlir::Type getType() const { return indexLabelType; }

private:
    mlir::Type indexLabelType;
};

class TensorTypeWrapper {
public:
    TensorTypeWrapper(mlir::Type tensorType) : tensorType(tensorType) {}

    mlir::ArrayAttr getIndices() {
        if (auto tensorWithIndicesType =
            tensorType
            .dyn_cast<mlir::tensor_network::TensorWithIndicesType>()) {
            return tensorWithIndicesType.getIndices();
        }
        return nullptr;
    }

    mlir::Type getType() const { return tensorType; }

private:
    mlir::Type tensorType;
};

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
        mlir::DialectRegistry registry;
        mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
        mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
        mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
        mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(
            registry);
        mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
        mlir::cf::registerBufferizableOpInterfaceExternalModels(registry);
        mlir::registerBuiltinDialectTranslation(registry);
        mlir::registerLLVMDialectTranslation(registry);


        mlir::ub::registerConvertUBToLLVMInterface(registry);
        mlir::registerConvertMemRefToLLVMInterface(registry);

        mlir::arith::registerConvertArithToLLVMInterface(registry);
        mlir::registerConvertComplexToLLVMInterface(registry);
        mlir::cf::registerConvertControlFlowToLLVMInterface(registry);
        mlir::func::registerAllExtensions(registry);
        mlir::registerConvertFuncToLLVMInterface(registry);
        mlir::index::registerConvertIndexToLLVMInterface(registry);
        mlir::registerConvertMathToLLVMInterface(registry);

        ctx_ = std::make_unique<mlir::MLIRContext>(registry);

        // ctx_ = std::make_unique<mlir::MLIRContext>();
        ctx_->getOrLoadDialect<mlir::tensor_network::TensorNetworkDialect>();
        ctx_->getOrLoadDialect<mlir::func::FuncDialect>();
        ctx_->getOrLoadDialect<mlir::arith::ArithDialect>();
        ctx_->getOrLoadDialect<mlir::linalg::LinalgDialect>();
        ctx_->getOrLoadDialect<mlir::bufferization::BufferizationDialect>();
        ctx_->getOrLoadDialect<mlir::memref::MemRefDialect>();
        ctx_->getOrLoadDialect<mlir::scf::SCFDialect>();
        ctx_->getOrLoadDialect<mlir::async::AsyncDialect>();

        builder_ = std::make_unique<mlir::OpBuilder>(ctx_.get());
        module_ = mlir::ModuleOp::create(builder_->getUnknownLoc());
        createFunction("main");
    }

    void createFunction(const std::string &name) {
        auto funcType = builder_->getFunctionType({}, {});
        auto func = builder_->create<mlir::func::FuncOp>(builder_->getUnknownLoc(),
                                                         name, funcType);

        func->setAttr("llvm.emit_c_interface", mlir::UnitAttr::get(builder_->getContext()));

        auto &entryBlock = *func.addEntryBlock();
        functions_.push_back(func);
        module_.push_back(func);


        builder_->setInsertionPointToEnd(&entryBlock);
        builder_->create<mlir::func::ReturnOp>(builder_->getUnknownLoc());

        builder_->setInsertionPointToStart(&entryBlock);
    }

    mlir::Operation *createIndexOp(int64_t size, const std::string &name) {
        auto sizeAttr = builder_->getI64IntegerAttr(size);
        auto nameAttr = builder_->getStringAttr(name);
        auto returnType = ::mlir::tensor_network::IndexLabelType::get(
            builder_->getContext(), sizeAttr, nameAttr);
        auto location = builder_->getUnknownLoc();
        auto indexOp = builder_->create<mlir::tensor_network::IndexOp>(
            location, returnType, sizeAttr, nameAttr);
        return indexOp;
    }

    mlir::Operation *createIndexOp(int64_t size) {
        std::string name = "index_" + std::to_string(nextIndexId_++);
        return createIndexOp(size, name);
    }

    mlir::Operation *createTensorOp(py::array array, py::args pyIndices) {
        auto shape = array.shape();
        auto data = static_cast<double *>(array.mutable_data());

        llvm::ArrayRef<long int> shapeRef(shape, array.ndim());

        auto tensorType =
            mlir::RankedTensorType::get(shapeRef, builder_->getF64Type());

        mlir::Location loc = builder_->getUnknownLoc();

        if (tensorType.getElementType().isF64()) {
            llvm::ArrayRef<double> dataRef(data, array.size());
            auto tensorValue = mlir::DenseElementsAttr::get(tensorType, dataRef);

            std::vector<mlir::Value> indexLabels;
            for (py::handle pyIndex : pyIndices) {
                auto indexWrapper = pyIndex.cast<OperationWrapper>();
                mlir::Operation *indexOp = indexWrapper.get();
                assert(indexOp->getNumResults() == 1 &&
                       "IndexOp should produce one result");
                mlir::Value indexValue = indexOp->getResult(0);
                indexLabels.push_back(indexValue);
            }

            if (indexLabels.size() != shapeRef.size()) {
                throw std::runtime_error(
                    "The number of indices does not match the "
                    "number of dimensions in the numpy array.");
            }

            for (size_t i = 0; i < indexLabels.size(); ++i) {
                auto indexOp = mlir::cast<mlir::tensor_network::IndexOp>(
                    indexLabels[i].getDefiningOp());
                if (static_cast<uint64_t>(indexOp.getSize()) !=
                    static_cast<uint64_t>(shapeRef[i])) {
                    throw std::runtime_error(
                        "The size of the index does not match the corresponding "
                        "dimension size in the numpy array.");
                }
            }

            // Convert mlir::Type to mlir::TypeAttr
            std::vector<mlir::Attribute> indexAttrs;
            for (const auto &indexLabel : indexLabels) {
                indexAttrs.push_back(mlir::TypeAttr::get(indexLabel.getType()));
            }

            auto tensorOp = builder_->create<mlir::tensor_network::TensorOp>(
                loc,
                mlir::tensor_network::TensorWithIndicesType::get(
                    ctx_.get(), tensorType, builder_->getArrayAttr(indexAttrs)),
                tensorValue, indexLabels);
            return tensorOp;
        } else {
            llvm::errs() << "Unsupported tensor element type: " << tensorType.getElementType() << "\n";
            return nullptr;
        }
    }

    mlir::Operation *createContractOp(OperationWrapper &lhs,
                                      OperationWrapper &rhs) {
        mlir::Value lhsValue = lhs.get()->getResult(0);
        mlir::Value rhsValue = rhs.get()->getResult(0);

        auto lhsTensorType =
            lhsValue.getType().cast<mlir::tensor_network::TensorWithIndicesType>();
        auto rhsTensorType =
            rhsValue.getType().cast<mlir::tensor_network::TensorWithIndicesType>();

        mlir::ArrayAttr lhsIndices = lhsTensorType.getIndices();
        mlir::ArrayAttr rhsIndices = rhsTensorType.getIndices();

        if (lhsIndices == nullptr || rhsIndices == nullptr) {
            throw std::runtime_error("Indices not found in tensor.");
        }

        // Compute unique indices and their sizes
        std::vector<mlir::Attribute> uniqueIndices;
        std::vector<int64_t> resultShape;

        auto addUniqueIndex = [&](mlir::Attribute indexAttr) {
            if (std::find(uniqueIndices.begin(), uniqueIndices.end(), indexAttr) ==
                uniqueIndices.end()) {
                uniqueIndices.push_back(indexAttr);
                auto indexLabelType = indexAttr.cast<mlir::TypeAttr>()
                    .getValue()
                    .cast<mlir::tensor_network::IndexLabelType>();
                resultShape.push_back(indexLabelType.getSize().getInt());
            }
        };

        for (auto lhsIndex : lhsIndices) {
            if (std::find(rhsIndices.begin(), rhsIndices.end(), lhsIndex) ==
                rhsIndices.end()) {
                addUniqueIndex(lhsIndex);
            }
        }
        for (auto rhsIndex : rhsIndices) {
            if (std::find(lhsIndices.begin(), lhsIndices.end(), rhsIndex) ==
                lhsIndices.end()) {
                addUniqueIndex(rhsIndex);
            }
        }

        // Create the result tensor type
        mlir::Type elementType =
            lhsTensorType.getTensorType().cast<mlir::ShapedType>().getElementType();
        auto resultTensorType =
            mlir::RankedTensorType::get(resultShape, elementType);

        mlir::ArrayAttr indexArrayAttr = builder_->getArrayAttr(uniqueIndices);

        mlir::Type returnType = mlir::tensor_network::TensorWithIndicesType::get(
            ctx_.get(), resultTensorType, indexArrayAttr);
        mlir::Location loc = builder_->getUnknownLoc();
        auto contractOp = builder_->create<mlir::tensor_network::ContractTensorsOp>(
            loc, returnType, lhsValue, rhsValue);
        return contractOp;
    }

    mlir::Operation *createContractMultipleTensorsOp(const std::vector<OperationWrapper> &tensors) {
        std::vector<mlir::Value> tensorValues;
        for (const auto &tensor : tensors) {
            tensorValues.push_back(const_cast<OperationWrapper&>(tensor).get()->getResult(0));
        }

        auto firstTensorType = tensorValues[0].getType().cast<mlir::tensor_network::TensorWithIndicesType>();

        // Determine all the free indices that are not shared
        std::vector<mlir::Attribute> allIndices;
        std::vector<mlir::Attribute> sharedIndices;
        std::vector<mlir::Attribute> freeIndices;
        std::vector<int64_t> resultShape;

        for (const auto &tensorValue : tensorValues) {
            auto tensorType = tensorValue.getType().cast<mlir::tensor_network::TensorWithIndicesType>();
            auto indices = tensorType.getIndices();
            for (auto index : indices) {
                if (std::find(allIndices.begin(), allIndices.end(), index) != allIndices.end()) {
                    if (std::find(sharedIndices.begin(), sharedIndices.end(), index) == sharedIndices.end()) {
                        sharedIndices.push_back(index);
                    }
                } else {
                    allIndices.push_back(index);
                }
            }
        }

        for (const auto &index : allIndices) {
            if (std::find(sharedIndices.begin(), sharedIndices.end(), index) == sharedIndices.end()) {
                freeIndices.push_back(index);
                auto indexType = index.cast<mlir::TypeAttr>().getValue().cast<mlir::tensor_network::IndexLabelType>();
                resultShape.push_back(indexType.getSize().getInt());
            }
        }

        auto elementType = firstTensorType.getTensorType().cast<mlir::ShapedType>().getElementType();
        auto resultTensorTypeWithShape = mlir::RankedTensorType::get(resultShape, elementType);

        auto resultType = mlir::tensor_network::TensorWithIndicesType::get(
            builder_->getContext(), 
            resultTensorTypeWithShape, 
            builder_->getArrayAttr(freeIndices));

        return builder_->create<mlir::tensor_network::ContractMultipleTensorsOp>(
            builder_->getUnknownLoc(),
            resultType,
            tensorValues);
    }

    mlir::ModuleOp getModule() { return module_; }

    void compile(bool useGreedyGrayKourtis, bool enableRankSimplification, bool enableSlicing, double grayKourtisAlpha) {
        
        loweringOptions.contractionStrategy = useGreedyGrayKourtis ? mlir::tensor_network::ContractionStrategy::GreedyGrayKourtis : mlir::tensor_network::ContractionStrategy::Greedy;
        loweringOptions.enableRankSimplification = enableRankSimplification;
        loweringOptions.enableSlicing = enableSlicing;
        loweringOptions.grayKourtisAlpha = grayKourtisAlpha;

        if (failed(lowerModule())) {
            llvm::errs() << "Failed to lower the module\n";
            throw std::runtime_error("Module compilation failed");
        }

        // Store shape information before lowering to LLVM
        auto mainFunc = module_.lookupSymbol<mlir::func::FuncOp>("main");
        if (!mainFunc) {
            throw std::runtime_error("Main function not found");
        }

        auto returnType = mainFunc.getResultTypes()[0];
        auto shapedType = returnType.dyn_cast<mlir::ShapedType>();
        if (!shapedType) {
            throw std::runtime_error("Return type is not a shaped type");
        }

        resultShape = shapedType.getShape();
        resultElementType = shapedType.getElementType();

        if (failed(lowerModuleToLLVM())) {
            llvm::errs() << "Failed to lower the module to LLVM\n";
            throw std::runtime_error("LLVM lowering failed");
        }
        isCompiled = true;
    }

    py::array run_compiled() {
        if (!isCompiled) {
            throw std::runtime_error("Module is not compiled. Call compile() first.");
        }

        int64_t totalSize = 1;
        for (auto dim : resultShape) {
            totalSize *= dim;
        }

        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();

        mlir::ExecutionEngineOptions engineOptions;
        engineOptions.transformer = mlir::makeOptimizingTransformer(
            /*optLevel=*/3, /*sizeLevel=*/0, /*targetMachine=*/nullptr);

        auto maybeEngine = mlir::ExecutionEngine::create(module_, engineOptions);
        if (!maybeEngine) {
            llvm::errs() << "Failed to create execution engine\n";
            return py::array();
        }
        std::unique_ptr<mlir::ExecutionEngine> engine = std::move(maybeEngine.get());

        std::vector<double> resultData(totalSize, 0.0);
        struct MemRefDescriptor {
            double* basePtr;
            double* data;
            int64_t offset;
            int64_t* sizes;
            int64_t* strides;
        } result;

        result.basePtr = resultData.data();
        result.data = resultData.data();
        result.offset = 0;

        std::vector<int64_t> shapeSizes(resultShape.begin(), resultShape.end());
        result.sizes = shapeSizes.data();

        std::vector<int64_t> strideValues(resultShape.size());
        int64_t stride = 1;
        for (int i = resultShape.size() - 1; i >= 0; --i) {
            strideValues[i] = stride;
            stride *= resultShape[i];
        }
        result.strides = strideValues.data();

        auto invocationResult = engine->invoke("main", &result);
        if (invocationResult) {
            llvm::errs() << "JIT invocation failed: " << invocationResult << "\n";
            return py::array();
        }

        std::vector<ssize_t> pyShape(resultShape.begin(), resultShape.end());
        auto resultArray = py::array_t<double>(pyShape);
        std::memcpy(resultArray.mutable_data(), result.data, totalSize * sizeof(double));
        return resultArray;
    }

    py::array run(bool useGreedyGrayKourtis, bool enableRankSimplification, bool enableSlicing, double grayKourtisAlpha) {

        loweringOptions.contractionStrategy = useGreedyGrayKourtis ? mlir::tensor_network::ContractionStrategy::GreedyGrayKourtis : mlir::tensor_network::ContractionStrategy::Greedy;
        loweringOptions.enableRankSimplification = enableRankSimplification;
        loweringOptions.enableSlicing = enableSlicing;
        loweringOptions.grayKourtisAlpha = grayKourtisAlpha;


        if (failed(lowerModule())) {
            llvm::errs() << "Failed to lower the module\n";
            return py::array();
        }

        auto mainFunc = module_.lookupSymbol<mlir::func::FuncOp>("main");
        if (!mainFunc) {
            llvm::errs() << "Main function not found\n";
            return py::array();
        }

        auto returnType = mainFunc.getResultTypes()[0];
        auto shapedType = returnType.dyn_cast<mlir::ShapedType>();
        if (!shapedType) {
            llvm::errs() << "Return type is not a shaped type\n";
            return py::array();
        }

        // Get the shape and element type
        auto shape = shapedType.getShape();

        // Calculate total size
        int64_t totalSize = 1;
        for (auto dim : shape) {
            totalSize *= dim;
        }

        // llvm::errs() << "Total size: " << totalSize << "\n";

        if (failed(lowerModuleToLLVM())) {
            llvm::errs() << "Failed to lower the module to LLVM\n";
            return py::array();
        }

        llvm::InitializeNativeTarget();
        llvm::InitializeNativeTargetAsmPrinter();

        mlir::ExecutionEngineOptions engineOptions;
        engineOptions.transformer = mlir::makeOptimizingTransformer(
            /*optLevel=*/3, /*sizeLevel=*/0, /*targetMachine=*/nullptr);

        auto maybeEngine = mlir::ExecutionEngine::create(module_, engineOptions);
        if (!maybeEngine) {
            llvm::errs() << "Failed to create execution engine\n";
            return py::array();
        }
        std::unique_ptr<mlir::ExecutionEngine> engine = std::move(maybeEngine.get());

        // Use a vector instead of raw pointer for automatic memory management
        std::vector<double> resultData(totalSize, 0.0);

        // Create MemRefDescriptor on the stack
        struct MemRefDescriptor {
            double* basePtr;
            double* data;
            int64_t offset;
            int64_t* sizes;
            int64_t* strides;
        } result;

        result.basePtr = resultData.data();
        result.data = resultData.data();
        result.offset = 0;

        std::vector<int64_t> shapeSizes(shape.begin(), shape.end());
        result.sizes = shapeSizes.data();

        // Calculate strides
        std::vector<int64_t> strideValues(shape.size());
        int64_t stride = 1;
        for (int i = shape.size() - 1; i >= 0; --i) {
            strideValues[i] = stride;
            stride *= shape[i];
        }
        result.strides = strideValues.data();

        // llvm::errs() << "Invoking JIT\n";
        auto invocationResult = engine->invoke("main", &result);
        if (invocationResult) {
            llvm::errs() << "JIT invocation failed: " << invocationResult << "\n";
            return py::array();
        }
        // llvm::errs() << "JIT invocation succeeded\n";

        // Create py::array with the correct shape
        std::vector<ssize_t> pyShape(shape.begin(), shape.end());
        auto resultArray = py::array_t<double>(pyShape);
        std::memcpy(resultArray.mutable_data(), result.data, totalSize * sizeof(double));
        return resultArray;
    }

private:
    std::unique_ptr<mlir::OpBuilder> builder_;
    mlir::ModuleOp module_;
    std::unique_ptr<mlir::MLIRContext> ctx_;
    int nextIndexId_ = 0;
    std::vector<mlir::func::FuncOp> functions_;

    bool isCompiled = false;
    mlir::ArrayRef<int64_t> resultShape;
    mlir::Type resultElementType;

    mlir::tensor_network::TensorNetworkNaiveLoweringOptions loweringOptions;

    mlir::LogicalResult lowerModule() {
        mlir::PassManager pm(ctx_.get());
        pm.addPass(mlir::createLocationSnapshotPass());
        pm.addPass(mlir::tensor_network::createTensorNetworkNaiveLoweringPass(loweringOptions));
        return pm.run(module_);
    }

    mlir::LogicalResult lowerModuleToLLVM() {
        mlir::PassManager pm(ctx_.get());
        // llvm::DebugFlag = "dialect-conversion";

        // pm.addPass(mlir::createConvertTensorToLinalgPass());
        // pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertVectorToSCFPass());
        // pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertLinalgToLoopsPass());
        //
        // mlir::bufferization::OneShotBufferizationOptions bufferizationOptions;
        // bufferizationOptions.bufferizeFunctionBoundaries = true;
        // pm.addPass(mlir::bufferization::createOneShotBufferizePass(bufferizationOptions));
        //
        //
        // pm.addPass(mlir::createAsyncToAsyncRuntimePass());
        // pm.addPass(mlir::createAsyncFuncToAsyncRuntimePass());
        // // pm.addPass(mlir::createAsyncRuntimeRefCountingPass());
        // pm.addPass(mlir::createAsyncRuntimeRefCountingOptPass());
        // // pm.addPass(mlir::createAsyncRuntimePolicyBasedRefCountingPass());
        //
        // pm.addPass(mlir::createLowerAffinePass());
        // pm.addPass(mlir::createConvertSCFToCFPass());
        // pm.addPass(mlir::createConvertLinalgToLoopsPass());
        //
        // pm.addPass(mlir::createConvertVectorToLLVMPass());
        // pm.addPass(mlir::createConvertMathToLLVMPass());
        // pm.addNestedPass<mlir::func::FuncOp>(mlir::createConvertMathToLLVMPass());
        //
        //
        // pm.addPass(mlir::createConvertControlFlowToLLVMPass());
        // pm.addPass(mlir::createArithToLLVMConversionPass());
        // pm.addPass(mlir::createConvertIndexToLLVMPass());
        // pm.addPass(mlir::memref::createExpandStridedMetadataPass());
        //
        //
        // pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
        // pm.addPass(mlir::createConvertAsyncToLLVMPass());
        //
        //
        // pm.addPass(mlir::createConvertFuncToLLVMPass());
        // pm.addPass(mlir::createReconcileUnrealizedCastsPass());
        //
        // pm.addPass(mlir::createConvertToLLVMPass());
        pm.addPass(mlir::createConvertTensorToLinalgPass());

        mlir::bufferization::OneShotBufferizationOptions bufferizationOptions;
        bufferizationOptions.bufferizeFunctionBoundaries = true;
        pm.addPass(mlir::bufferization::createOneShotBufferizePass(bufferizationOptions));

        pm.addPass(mlir::createConvertLinalgToLoopsPass());
        pm.addPass(mlir::createLowerAffinePass());
        pm.addPass(mlir::createConvertSCFToCFPass());
        pm.addPass(mlir::createConvertControlFlowToLLVMPass());
        pm.addPass(mlir::createConvertFuncToLLVMPass());
        pm.addPass(mlir::createArithToLLVMConversionPass());
        pm.addPass(mlir::createConvertMathToLLVMPass());
        pm.addPass(mlir::createConvertIndexToLLVMPass());
        pm.addPass(mlir::createConvertVectorToLLVMPass());
        pm.addPass(mlir::memref::createExpandStridedMetadataPass());
        pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
        pm.addPass(mlir::createReconcileUnrealizedCastsPass());
        return pm.run(module_);
    }
};

PYBIND11_MODULE(tensor_network_ext, m) {
    m.doc() = "TensorNetwork Dialect bindings";

    py::class_<OperationWrapper>(m, "Operation")
        .def(py::init<mlir::Operation *>())
        .def("get", &OperationWrapper::get)
        .def("dump", [](OperationWrapper &self) { self.get()->dump(); });

    py::class_<ModuleManager>(m, "ModuleManager")
        .def(py::init<>())
        .def("Index",
             [](ModuleManager &self, int64_t size, const std::string &name) {
             return OperationWrapper(self.createIndexOp(size, name));
             })
        .def("Index",
             [](ModuleManager &self, int64_t size) {
             return OperationWrapper(self.createIndexOp(size));
             })
        .def("Tensor",
             [](ModuleManager &self, py::array array, py::args indices) {
             return OperationWrapper(self.createTensorOp(array, indices));
             })
        .def("contract",
             [](ModuleManager &self, OperationWrapper &lhs,
                OperationWrapper &rhs) {
             return OperationWrapper(self.createContractOp(lhs, rhs));
             })
        .def("dump", [](ModuleManager &self) { self.getModule().dump(); })
        .def("contract_multiple", [](ModuleManager &self, py::args tensors) {
            std::vector<OperationWrapper> tensorOps;
            for (auto tensor : tensors) {
                tensorOps.push_back(tensor.cast<OperationWrapper>());
            }
            return OperationWrapper(self.createContractMultipleTensorsOp(tensorOps));
        })
        // .def("run", [](ModuleManager &self) { return self.run(); })
        // .def("compile", &ModuleManager::compile)
        .def("compile", &ModuleManager::compile,
             py::arg("use_greedy_gray_kourtis") = false,
             py::arg("enable_rank_simplification") = true,
             py::arg("enable_slicing") = false,
             py::arg("gray_kourtis_alpha") = 1.0)
        .def("run", &ModuleManager::run,
             py::arg("use_greedy_gray_kourtis") = false,
             py::arg("enable_rank_simplification") = true,
             py::arg("enable_slicing") = false,
             py::arg("gray_kourtis_alpha") = 1.0)
        .def("run_compiled", &ModuleManager::run_compiled)
        .def("pre_compile", [](ModuleManager &self) { return; })
        .def("load", [](ModuleManager &self, const std::string &filename) { return; });
}
