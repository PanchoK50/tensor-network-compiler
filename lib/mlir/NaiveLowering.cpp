#include "TensorNetwork/Passes.h"
#include "TensorNetwork/TensorNetworkDialect.h"
#include "mlir/IR/Verifier.h"
#include "TensorNetwork/TensorNetworkOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "TensorNetwork/TensorNetworkTypes.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/FoldUtils.h"
#include "mlir/Transforms/RegionUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Types.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include <vector>
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/Bufferize.h"

#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// TensorNetworkToNaive RewritePatterns
//===----------------------------------------------------------------------===//

struct TensorDeclOpLowering : public OpRewritePattern<tensor_network::TensorDeclOp> {
    using OpRewritePattern<tensor_network::TensorDeclOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(tensor_network::TensorDeclOp op,
                                  PatternRewriter &rewriter) const final {

        /*
         *  In the end, we want that the declaration of the tensor creates
         *  a space in memory where the values will be stored.
         *  For this, we start at the highest abstraction level and
         *  utilize the already existing lowerings to the memref dialect.
         *
         *  We will use the tensor_with_indices type to propagate the indice and at the end eliminate the type
        */

        // This operation only takes the indices to determine the shape
        // since it doesn't have any values yet.

        // Get the shape of the tensor

        // TODO: determine if the best way to determine the shape is in the
        // front- or backend. For now doing it here.

        //Extract all the indices
        SmallVector<int64_t, 4> shape;
        std::vector<mlir::Attribute> indices;

        for (auto index : op.getIndices()) {
            auto indexOp = dyn_cast<tensor_network::IndexOp>(index.getDefiningOp());
            if (!indexOp) {
                return rewriter.notifyMatchFailure(op, "IndexOp not found");
            }

            // Get the size of the index
            auto size = indexOp.getSize();
            shape.push_back(size);

            mlir::Value indexValue = indexOp.getResult();
            mlir::Attribute indexAttr = mlir::TypeAttr::get(indexValue.getType()); 

            indices.push_back(indexAttr);
        }

        // Create empty tensor with the shape using tensor.empty()
        auto tensorType = RankedTensorType::get(shape, rewriter.getF64Type());
        auto tensor = rewriter.create<tensor::EmptyOp>(op.getLoc(), shape, tensorType);

        auto resultIndicesArrayAttr = rewriter.getArrayAttr(indices);
        auto tensorWithIndicesType = tensor_network::TensorWithIndicesType::get(rewriter.getContext(), tensorType, resultIndicesArrayAttr);
        auto tensorWithValuesOp = rewriter.create<tensor_network::TensorFromValueOp>(op.getLoc(), tensorWithIndicesType, tensor);

        rewriter.replaceOp(op, tensorWithValuesOp.getResult());

        return success();
    }
};

struct TensorOpLowering : public OpRewritePattern<tensor_network::TensorOp> {
    using OpRewritePattern<tensor_network::TensorOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(tensor_network::TensorOp op,
                                  PatternRewriter &rewriter) const final {
        // Extract the shape from the indices
        SmallVector<int64_t, 4> shape;
        std::vector<mlir::Attribute> indices;

        for (auto index : op.getIndices()) {
            auto indexOp = dyn_cast<tensor_network::IndexOp>(index.getDefiningOp());
            if (!indexOp) {
                return rewriter.notifyMatchFailure(op, "IndexOp not found");
            }

            auto size = indexOp.getSize();
            shape.push_back(size);

            mlir::Value indexValue = indexOp.getResult();
            mlir::Attribute indexAttr = mlir::TypeAttr::get(indexValue.getType()); 
            indices.push_back(indexAttr);
        }

        // Get the values attribute
        auto valuesAttr = op.getValue();
        if (!valuesAttr) {
            return rewriter.notifyMatchFailure(op, "Values attribute not found");
        }

        // Create the tensor type
        auto elementType = rewriter.getF64Type(); // Assuming double precision floats
        auto tensorType = RankedTensorType::get(shape, elementType);

        // Create a constant tensor directly
        auto tensor = rewriter.create<arith::ConstantOp>(
            op.getLoc(), 
            tensorType, 
            valuesAttr);

        // Create the TensorWithIndicesType
        auto resultIndicesArrayAttr = rewriter.getArrayAttr(indices);
        auto tensorWithIndicesType = tensor_network::TensorWithIndicesType::get(
            rewriter.getContext(), tensorType, resultIndicesArrayAttr);

        // Directly pass the constant tensor to TensorWithValuesOp
        auto tensorWithValuesOp = rewriter.create<tensor_network::TensorFromValueOp>(
            op.getLoc(), tensorWithIndicesType, tensor);

        // Replace the original op with the new one
        rewriter.replaceOp(op, tensorWithValuesOp.getResult());

        return success();
    }
};

struct IndexOpLowering : public OpRewritePattern<tensor_network::IndexOp> {
    using OpRewritePattern<tensor_network::IndexOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(tensor_network::IndexOp op,
                                  PatternRewriter &rewriter) const final {

        return success();
    }
};

struct RemoveTensorFromValueOp : public OpRewritePattern<tensor_network::TensorFromValueOp> {
    using OpRewritePattern<tensor_network::TensorFromValueOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(tensor_network::TensorFromValueOp op,
                                  PatternRewriter &rewriter) const override {

        // Just remove the TensorFromValueOp
        rewriter.replaceOp(op, op.getOperand());

        return success();
    }
};

struct ContractOpLowering : public OpRewritePattern<tensor_network::ContractTensorsOp> {
    using OpRewritePattern<tensor_network::ContractTensorsOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(tensor_network::ContractTensorsOp op,
                                  PatternRewriter &rewriter) const final {

        Value lhs = op.getLhs();
        Value rhs = op.getRhs();

        //Check if all the arguments are created via the TensorFromValueOp
        if (!lhs.getDefiningOp<tensor_network::TensorFromValueOp>() || !rhs.getDefiningOp<tensor_network::TensorFromValueOp>()) {
            return failure();
        }

        auto lhsType = lhs.getType().dyn_cast<tensor_network::TensorWithIndicesType>();
        auto rhsType = rhs.getType().dyn_cast<tensor_network::TensorWithIndicesType>();

        if (!lhsType || !rhsType) {
            return failure();
        }

        auto lhsIndices = lhsType.getIndices();
        auto rhsIndices = rhsType.getIndices();

        Value lhsTensor = extractTensorValue(rewriter, lhs);
        Value rhsTensor = extractTensorValue(rewriter, rhs);

        if (!lhsTensor || !rhsTensor) {
            return failure();
        }

        SmallVector<mlir::Attribute> uniqueIndices;
        for (auto lhsIndex : lhsIndices) {
            if (std::find(uniqueIndices.begin(), uniqueIndices.end(), lhsIndex) == uniqueIndices.end()) {
                uniqueIndices.push_back(lhsIndex);
            }
        }
        for (auto rhsIndex : rhsIndices) {
            if (std::find(uniqueIndices.begin(), uniqueIndices.end(), rhsIndex) == uniqueIndices.end()) {
                uniqueIndices.push_back(rhsIndex);
            }
        }

        SmallVector<mlir::Attribute> commonIndices;
        for (auto lhsIndex : lhsIndices) {
            if (std::find(rhsIndices.begin(), rhsIndices.end(), lhsIndex) != rhsIndices.end()) {
                commonIndices.push_back(lhsIndex);
            }
        }

        if (commonIndices.empty()) {
            return failure();
        }

        auto resultType = op.getResult().getType().cast<tensor_network::TensorWithIndicesType>().getTensorType();

        auto zeroConstant = rewriter.create<mlir::arith::ConstantOp>(op.getLoc(), resultType, rewriter.getZeroAttr(resultType));

        SmallVector<AffineMap, 3> indexingMaps;

        SmallVector<AffineExpr, 4> lhsExprs;
        for (unsigned i = 0; i < uniqueIndices.size(); ++i) {
            if (std::find(lhsIndices.begin(), lhsIndices.end(), uniqueIndices[i]) != lhsIndices.end()) {
                lhsExprs.push_back(rewriter.getAffineDimExpr(i));
            }
        }
        indexingMaps.push_back(AffineMap::get(uniqueIndices.size(), 0, lhsExprs, rewriter.getContext()));

        SmallVector<AffineExpr, 4> rhsExprs;
        for (unsigned i = 0; i < uniqueIndices.size(); ++i) {
            if (std::find(rhsIndices.begin(), rhsIndices.end(), uniqueIndices[i]) != rhsIndices.end()) {
                rhsExprs.push_back(rewriter.getAffineDimExpr(i));
            }
        }
        indexingMaps.push_back(AffineMap::get(uniqueIndices.size(), 0, rhsExprs, rewriter.getContext()));

        SmallVector<AffineExpr, 4> resultExprs;
        for (unsigned i = 0; i < uniqueIndices.size(); ++i) {
            if (std::find(commonIndices.begin(), commonIndices.end(), uniqueIndices[i]) == commonIndices.end()) {
                resultExprs.push_back(rewriter.getAffineDimExpr(i));
            }
        }
        indexingMaps.push_back(AffineMap::get(uniqueIndices.size(), 0, resultExprs, rewriter.getContext()));

        SmallVector<utils::IteratorType> iteratorTypes;
        for (unsigned i = 0; i < uniqueIndices.size(); ++i) {
            if (std::find(commonIndices.begin(), commonIndices.end(), uniqueIndices[i]) != commonIndices.end()) {
                iteratorTypes.push_back(utils::IteratorType::reduction);
            } else {
                iteratorTypes.push_back(utils::IteratorType::parallel);
            }
        }


        auto genericOp = rewriter.create<linalg::GenericOp>(
            op.getLoc(), TypeRange{resultType}, ValueRange{lhsTensor, rhsTensor}, ValueRange{zeroConstant.getResult()},
            indexingMaps, iteratorTypes,
            [](OpBuilder &builder, Location loc, ValueRange args) {
                auto mulOp = builder.create<mlir::arith::MulFOp>(loc, args[0], args[1]);
                auto addOp = builder.create<mlir::arith::AddFOp>(loc, mulOp, args[2]);
                builder.create<linalg::YieldOp>(loc, addOp.getResult());
            });

        // now we need to put the result from the generic op into a tensorFromValueOp
        auto resultTensor = genericOp.getResult(0);
        auto newTensorFromValueOp = rewriter.create<tensor_network::TensorFromValueOp>(op.getLoc(), op.getResult().getType(), resultTensor);

        rewriter.replaceOp(op, newTensorFromValueOp.getResult());

        return success(); 
    }

private:
    // Helper function to extract tensor value based on operation type
    Value extractTensorValue(PatternRewriter &rewriter, Value v) const {
        if (auto tensorFromValueOp = v.getDefiningOp<tensor_network::TensorFromValueOp>()) {
            // Extract the underlying tensor from TensorFromValueOp
            return tensorFromValueOp.getOperand();
        }
        return nullptr;
    }

};


struct ContractMultipleTensorsOpLowering : public OpRewritePattern<tensor_network::ContractMultipleTensorsOp> {
    using OpRewritePattern<tensor_network::ContractMultipleTensorsOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(tensor_network::ContractMultipleTensorsOp op,
                                  PatternRewriter &rewriter) const final {




        return success();
    }
};

struct ReturnFinalTensorPattern : public OpRewritePattern<mlir::func::FuncOp> {
    using OpRewritePattern<mlir::func::FuncOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(mlir::func::FuncOp op,
                                  PatternRewriter &rewriter) const final {


        return mlir::success();
    }
};

struct WriteFinalTensorPattern : public OpRewritePattern<mlir::func::FuncOp> {
    using OpRewritePattern<mlir::func::FuncOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(mlir::func::FuncOp op,
                                  PatternRewriter &rewriter) const final {

        llvm::errs() << "Matching FuncOp: " << op.getName() << "\n";

        // Check if the operation has already been rewritten
        if (op->hasAttr("rewritten")) {
            return failure();
        }

        // Check if the main function is the one we want to modify
        if (op.getName() != "main") {
            llvm::errs() << "Only modifying the main function\n";
            return failure();
        }

        // Find the last tensorFromValueOp in the main function
        mlir::tensor_network::TensorFromValueOp lastTensorFromValueOp;
        for (auto &innerOp : llvm::reverse(op.getBody().front())) {
            if (auto tensorOp = dyn_cast<mlir::tensor_network::TensorFromValueOp>(innerOp)) {
                lastTensorFromValueOp = tensorOp;
                break;
            }
        }

        if (!lastTensorFromValueOp) {
            llvm::errs() << "No TensorFromValueOp found in the main function\n";
            return mlir::failure();
        }

        // Get the actual tensor from the argument of the TensorFromValueOp
        mlir::Value tensor = lastTensorFromValueOp.getOperand();
        if (!tensor) {
            llvm::errs() << "Invalid tensor operand\n";
            return mlir::failure();
        }

        // Find the return operation
        auto returnOp = op.getBody().front().getTerminator();
        if (!returnOp) {
            llvm::errs() << "No return operation found in the function\n";
            return mlir::failure();
        }

        // Create a new builder at the location of the return operation
        OpBuilder::InsertionGuard guard(rewriter);
        rewriter.setInsertionPoint(returnOp);

        // Replace the return operation to return the computed tensor directly
        rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(returnOp, tensor);

        // Modify the function type to reflect the new return type and no arguments
        auto funcType = rewriter.getFunctionType({}, tensor.getType());
        op.setType(funcType);

        op->setAttr("rewritten", rewriter.getUnitAttr());

        return mlir::success();
    }
};


//===----------------------------------------------------------------------===//
// TensorNetworkToNaive Lowering Pass
//===----------------------------------------------------------------------===//

namespace {
struct TensorNetworkNaiveLoweringPass
: public PassWrapper<TensorNetworkNaiveLoweringPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TensorNetworkNaiveLoweringPass)

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<mlir::linalg::LinalgDialect, mlir::func::FuncDialect,
            mlir::memref::MemRefDialect, mlir::arith::ArithDialect,
            mlir::tensor::TensorDialect, mlir::bufferization::BufferizationDialect>();
        mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
        mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
    }

    void runOnOperation() final;
};
}  // namespace

void TensorNetworkNaiveLoweringPass::runOnOperation() {
    auto module = getOperation();
    auto *context = &getContext();

    ConversionTarget target(*context);
    target.addLegalDialect<mlir::func::FuncDialect, mlir::arith::ArithDialect,
        mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect,
        mlir::memref::MemRefDialect, mlir::bufferization::BufferizationDialect>();

    // Step 1: Apply the lowering patterns
    {
        llvm::errs() << "Step 1: Apply the lowering patterns\n";

        RewritePatternSet loweringPatterns(&getContext());
        loweringPatterns.add<TensorDeclOpLowering, TensorOpLowering, IndexOpLowering, 
            ContractOpLowering>(&getContext());

        if (failed(applyPatternsAndFoldGreedily(module, std::move(loweringPatterns)))) {
            signalPassFailure();
            return;
        }

        module.dump();

        if (failed(mlir::verify(module))) {
            llvm::errs() << "Error in module verification\n";
            signalPassFailure();
            return;
        }
    }

    // Step 2: Return the final tensor
    {

        RewritePatternSet writeFinalTensorPatterns(&getContext());
        writeFinalTensorPatterns.add<WriteFinalTensorPattern>(&getContext());

        if (failed(applyPatternsAndFoldGreedily(module, std::move(writeFinalTensorPatterns)))) {
            llvm::errs() << "Failed to apply the WriteFinalTensorPattern\n";
            signalPassFailure();
            return;
        }

        module.dump();

        if (failed(mlir::verify(module))) {
            llvm::errs() << "Error in module verification\n";
            signalPassFailure();
            return;
        }
    }

    target.addIllegalDialect<tensor_network::TensorNetworkDialect>();

    // Step 3: Apply the RemoveTensorFromValuesOp pattern
    {
        llvm::errs() << "Step 3: Apply the RemoveTensorFromValuesOp pattern\n";

        RewritePatternSet removalPatterns(&getContext());
        removalPatterns.add<RemoveTensorFromValueOp>(&getContext());

        if (failed(applyPartialConversion(module, target, std::move(removalPatterns)))) {
            signalPassFailure();
            return;
        }

        module.dump();
        if (failed(mlir::verify(module))) {
            llvm::errs() << "Error in module verification\n";
            signalPassFailure();
            return;
        }
    }
}


std::unique_ptr<mlir::Pass> mlir::tensor_network::createTensorNetworkNaiveLoweringPass() {
    return std::make_unique<TensorNetworkNaiveLoweringPass>();
}
