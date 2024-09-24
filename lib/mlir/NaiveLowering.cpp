#include "TensorNetwork/Passes.h"
#include "TensorNetwork/TensorNetworkDialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Dialect/Async/Transforms.h"
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
#include "mlir/Dialect/Async/IR/Async.h"
#include "mlir/Dialect/Async/IR/AsyncTypes.h"
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
#include "mlir/Transforms/DialectConversion.h"

#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// TensorNetworkToNaive RewritePatterns
//===----------------------------------------------------------------------===//

class AsyncYieldOpLowering : public OpConversionPattern<async::YieldOp> {
public:
    using OpConversionPattern<async::YieldOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(async::YieldOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {

        auto yieldOp = rewriter.create<async::YieldOp>(op.getLoc(), adaptor.getOperands());
        rewriter.replaceOp(op, yieldOp);
        return success();
    }
};

class AsyncAwaitOpLowering : public OpConversionPattern<async::AwaitOp> {
public:
    using OpConversionPattern<async::AwaitOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(async::AwaitOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {

        auto awaitOp = rewriter.create<async::AwaitOp>(op.getLoc(), adaptor.getOperand());
        rewriter.replaceOp(op, awaitOp.getResult());
        return success();
    }
};

// class AsyncExecuteOpLowering : public OpConversionPattern<async::ExecuteOp> {
// public:
//     using OpConversionPattern<async::ExecuteOp>::OpConversionPattern;
//     
//     LogicalResult matchAndRewrite(async::ExecuteOp op, OpAdaptor adaptor,
//                                   ConversionPatternRewriter &rewriter) const override {
//
//     auto newOp =
//         cast<async::ExecuteOp>(rewriter.cloneWithoutRegions(*op.getOperation()));
//     rewriter.inlineRegionBefore(op.getRegion(), newOp.getRegion(),
//                                 newOp.getRegion().end());
//
//     // Set operands and update block argument and result types.
//     newOp->setOperands(adaptor.getOperands());
//     if (failed(rewriter.convertRegionTypes(&newOp.getRegion(), *typeConverter)))
//       return failure();
//     for (auto result : newOp.getResults())
//       result.setType(typeConverter->convertType(result.getType()));
//
//     rewriter.replaceOp(op, newOp.getResults());
//     return success();
//     }
// };

class TensorDeclOpLowering : public OpConversionPattern<tensor_network::TensorDeclOp> {
public:

    using OpConversionPattern<tensor_network::TensorDeclOp>::OpConversionPattern;
    using OpAdaptor = typename tensor_network::TensorDeclOp::Adaptor;

    LogicalResult matchAndRewrite(tensor_network::TensorDeclOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {

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

class TensorOpLowering : public OpConversionPattern<tensor_network::TensorOp> {
public:

    using OpConversionPattern<tensor_network::TensorOp>::OpConversionPattern;
    using OpAdaptor = typename tensor_network::TensorOp::Adaptor;

    LogicalResult matchAndRewrite(tensor_network::TensorOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {

        // llvm::errs() << "Matching TensorOp\n";

        // Get the values attribute
        auto valuesAttr = op.getValue();
        if (!valuesAttr) {
            return rewriter.notifyMatchFailure(op, "Values attribute not found");
        }

        auto tensorType = op.getType().getTensorType();

        // Create a constant tensor directly
        auto tensor = rewriter.create<arith::ConstantOp>(
            op.getLoc(), 
            tensorType, 
            valuesAttr); 

        // Replace the original op with the new one
        rewriter.replaceOp(op, tensor);

        return success();
    }
};

class IndexOpLowering : public OpConversionPattern<tensor_network::IndexOp> {
public:

    using OpConversionPattern<tensor_network::IndexOp>::OpConversionPattern;
    using OpAdaptor = typename tensor_network::IndexOp::Adaptor;

    LogicalResult matchAndRewrite(tensor_network::IndexOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {


        rewriter.replaceOp(op, rewriter.create<arith::ConstantOp>(
            op.getLoc(), rewriter.getI64Type(), rewriter.getI64IntegerAttr(op.getSize())));

        return success();
    }
};

class RemoveTensorFromValueOp : public OpConversionPattern <tensor_network::TensorFromValueOp> {
public:

    using OpConversionPattern<tensor_network::TensorFromValueOp>::OpConversionPattern;
    using OpAdaptor = typename tensor_network::TensorFromValueOp::Adaptor;

    LogicalResult matchAndRewrite(tensor_network::TensorFromValueOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {

        // Just remove the TensorFromValueOp
        rewriter.replaceOp(op, op.getOperand());

        return success();
    }
};

// struct RemoveTensorFromValueOp : public OpRewritePattern<tensor_network::TensorFromValueOp> {
//     using OpRewritePattern<tensor_network::TensorFromValueOp>::OpRewritePattern;
//
//     LogicalResult matchAndRewrite(tensor_network::TensorFromValueOp op,
//                                   PatternRewriter &rewriter) const override {
//
//         // Just remove the TensorFromValueOp
//         rewriter.replaceOp(op, op.getOperand());
//
//         return success();
//     }
// };

class FuncOpLowering : public OpConversionPattern<mlir::func::FuncOp> {
public:
    using OpConversionPattern<mlir::func::FuncOp>::OpConversionPattern;
    using OpAdaptor = typename mlir::func::FuncOp::Adaptor;

    LogicalResult matchAndRewrite(mlir::func::FuncOp op, OpAdaptor adaptor,
                            ConversionPatternRewriter &rewriter) const override {

        //Get the return type of the function and change it to return a tensor

        auto tensorType = op.getResultTypes()[0].cast<tensor_network::TensorWithIndicesType>().getTensorType();
        auto funcType = rewriter.getFunctionType({}, tensorType); 
        op.setType(funcType);

        return success();
    }
};

class ContractOpLowering : public OpConversionPattern<tensor_network::ContractTensorsOp> {
public:

    using OpConversionPattern<tensor_network::ContractTensorsOp>::OpConversionPattern;
    using OpAdaptor = typename tensor_network::ContractTensorsOp::Adaptor;

    LogicalResult matchAndRewrite(tensor_network::ContractTensorsOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {

        // llvm::errs() << "Matching ContractTensorsOp\n";

        Value lhs = op.getLhs();
        Value rhs = op.getRhs();

        auto lhsType = lhs.getType().dyn_cast<tensor_network::TensorWithIndicesType>();
        auto rhsType = rhs.getType().dyn_cast<tensor_network::TensorWithIndicesType>();

        if (!lhsType || !rhsType) {
            llvm::errs() << "Failed to cast to TensorWithIndicesType\n";
            return failure();
        }

        auto lhsIndices = lhsType.getIndices();
        auto rhsIndices = rhsType.getIndices();

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
            llvm::errs() << "No common indices found\n";
            return failure();
        }

        auto resultTensorType = op.getResult().getType().cast<tensor_network::TensorWithIndicesType>().getTensorType();
        auto zeroConstant = rewriter.create<mlir::arith::ConstantOp>(op.getLoc(), op.getType().getTensorType(), rewriter.getZeroAttr(resultTensorType));

        SmallVector<AffineMap, 3> indexingMaps;

        // Create indexing map for left-hand side tensor
        SmallVector<AffineExpr, 4> lhsExprs;
        for (auto lhsIndex : lhsIndices) {
            auto it = std::find(uniqueIndices.begin(), uniqueIndices.end(), lhsIndex);
            if (it != uniqueIndices.end()) {
                lhsExprs.push_back(rewriter.getAffineDimExpr(std::distance(uniqueIndices.begin(), it)));
            }
        }
        indexingMaps.push_back(AffineMap::get(uniqueIndices.size(), 0, lhsExprs, rewriter.getContext()));

        // Create indexing map for right-hand side tensor
        SmallVector<AffineExpr, 4> rhsExprs;
        for (auto rhsIndex : rhsIndices) {
            auto it = std::find(uniqueIndices.begin(), uniqueIndices.end(), rhsIndex);
            if (it != uniqueIndices.end()) {
                rhsExprs.push_back(rewriter.getAffineDimExpr(std::distance(uniqueIndices.begin(), it)));
            }
        }
        indexingMaps.push_back(AffineMap::get(uniqueIndices.size(), 0, rhsExprs, rewriter.getContext()));

        // Create indexing map for result tensor
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
            op.getLoc(), TypeRange{op.getType().getTensorType()}, ValueRange{adaptor.getLhs(), adaptor.getRhs()}, ValueRange{zeroConstant.getResult()},
            indexingMaps, iteratorTypes,
            [](OpBuilder &builder, Location loc, ValueRange args) {
                auto mulOp = builder.create<mlir::arith::MulFOp>(loc, args[0], args[1]);
                auto addOp = builder.create<mlir::arith::AddFOp>(loc, mulOp, args[2]);
                builder.create<linalg::YieldOp>(loc, addOp.getResult());
            });

        rewriter.replaceOp(op, genericOp.getResult(0));

        return success();
    }
};


class AddOpLowering : public OpConversionPattern<tensor_network::AddOp> {
public:

    using OpConversionPattern<tensor_network::AddOp>::OpConversionPattern;
    using OpAdaptor = typename tensor_network::AddOp::Adaptor;

    LogicalResult matchAndRewrite(tensor_network::AddOp op, OpAdaptor adaptor,
                                  ConversionPatternRewriter &rewriter) const override {

        Value lhs = op.getLhs();
        Value rhs = op.getRhs();

        auto lhsType = lhs.getType().dyn_cast<tensor_network::TensorWithIndicesType>();
        auto rhsType = rhs.getType().dyn_cast<tensor_network::TensorWithIndicesType>();

        if (!lhsType || !rhsType) {
            llvm::errs() << "Failed to cast to TensorWithIndicesType\n";
            return failure();
        }

        auto lhsIndices = lhsType.getIndices();
        auto rhsIndices = rhsType.getIndices();

        // Check if indices match exactly
        if (lhsIndices != rhsIndices) {
            llvm::errs() << "Indices do not match\n";
            return failure();
        }

        // Check if dimensions match
        auto lhsTensorType = lhsType.getTensorType().dyn_cast<RankedTensorType>();
        auto rhsTensorType = rhsType.getTensorType().dyn_cast<RankedTensorType>();
        if (!lhsTensorType || !rhsTensorType || lhsTensorType.getShape() != rhsTensorType.getShape()) {
            llvm::errs() << "Tensor dimensions do not match\n";
            return failure();
        }

        auto zeroConstant = rewriter.create<mlir::arith::ConstantOp>(
            op.getLoc(), op.getType().getTensorType(), rewriter.getZeroAttr(op.getType().getTensorType())
        );

        auto rank = lhsTensorType.getRank();

        auto genericOp = rewriter.create<linalg::GenericOp>(
            op.getLoc(),
            TypeRange{op.getType().getTensorType()},
            ValueRange{adaptor.getLhs(), adaptor.getRhs()},
            ValueRange{zeroConstant},
            ArrayRef<AffineMap>{
                rewriter.getMultiDimIdentityMap(rank),
                rewriter.getMultiDimIdentityMap(rank),
                rewriter.getMultiDimIdentityMap(rank)
            },
            getNParallelLoopsAttrs(rank),
            [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
                Value add = nestedBuilder.create<arith::AddFOp>(nestedLoc, args[0], args[1]);
                nestedBuilder.create<linalg::YieldOp>(nestedLoc, add);
            }
        );

        rewriter.replaceOp(op, genericOp.getResult(0));
        return success();
    }

private:
    // Helper function to get N parallel loops attribute
    SmallVector<utils::IteratorType, 4> getNParallelLoopsAttrs(unsigned nLoops) const {
        return SmallVector<utils::IteratorType, 4>(nLoops, utils::IteratorType::parallel);
    }
};

struct RankSimplification : public OpRewritePattern<tensor_network::ContractMultipleTensorsOp> {
    using OpRewritePattern<tensor_network::ContractMultipleTensorsOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(tensor_network::ContractMultipleTensorsOp op,
                                  PatternRewriter &rewriter) const override {
        SmallVector<Value, 4> tensors(op.getTensors().begin(), op.getTensors().end());
        bool changed = false;

        // Iterate until no more simplifications can be made
        while (true) {
            bool simplificationMade = false;

            for (size_t i = 0; i < tensors.size(); ++i) {
                auto tensorType = tensors[i].getType().cast<tensor_network::TensorWithIndicesType>();

                for (size_t j = 0; j < tensors.size(); ++j) {
                    if (i == j) continue;

                    if (canAbsorb(tensors[i], tensors[j])) {
                        auto newTensor = absorbTensors(rewriter, op.getLoc(), tensors[i], tensors[j]);
                        tensors[j] = newTensor;
                        tensors.erase(tensors.begin() + i);
                        simplificationMade = true;
                        changed = true;
                        break;
                    }
                }

                if (simplificationMade) break;
            }

            if (!simplificationMade) break;
        }

        if (changed) {
            // Create a new ContractMultipleTensorsOp with the simplified tensors
            auto newOp = rewriter.create<tensor_network::ContractMultipleTensorsOp>(
                op.getLoc(), op.getType(), tensors);
            rewriter.replaceOp(op, newOp.getResult());
            return success();
        }

        return failure();
    }

private:
    bool canAbsorb(Value tensor1, Value tensor2) const {
        auto type1 = tensor1.getType().cast<tensor_network::TensorWithIndicesType>();
        auto type2 = tensor2.getType().cast<tensor_network::TensorWithIndicesType>();
        auto indices1 = type1.getIndices();
        auto indices2 = type2.getIndices();

        int commonIndices = 0;
        for (auto index : indices1) {
            if (llvm::is_contained(indices2, index)) {
                commonIndices++;
            }
        }

        if (commonIndices == 0) return false;

        int rank1 = type1.getTensorType().cast<RankedTensorType>().getRank();
        int rank2 = type2.getTensorType().cast<RankedTensorType>().getRank();

        // Calculate the rank of the result if these tensors were contracted
        int newRank = rank1 + rank2 - 2 * commonIndices;

        // The absorption is allowed if the new rank is not larger than the larger of the two input ranks
        return newRank <= std::max(rank1, rank2);
    }

    Value absorbTensors(PatternRewriter &rewriter, Location loc, Value tensor1, Value tensor2) const {
        // Create a ContractTensorsOp to perform the absorption
        auto contractOp = rewriter.create<tensor_network::ContractTensorsOp>(
            loc, 
            determineContractedTensorType(tensor1, tensor2, rewriter),
            tensor1, tensor2);

        return contractOp.getResult();
    }

    tensor_network::TensorWithIndicesType determineContractedTensorType(Value lhs, Value rhs, PatternRewriter &rewriter) const {
        auto lhsType = lhs.getType().cast<tensor_network::TensorWithIndicesType>();
        auto rhsType = rhs.getType().cast<tensor_network::TensorWithIndicesType>();
        auto lhsIndices = lhsType.getIndices();
        auto rhsIndices = rhsType.getIndices();

        SmallVector<Attribute, 4> resultIndices;
        SmallVector<int64_t, 4> resultShape;

        for (auto index : lhsIndices) {
            if (!llvm::is_contained(rhsIndices, index)) {
                resultIndices.push_back(index);
                resultShape.push_back(index.cast<TypeAttr>().getValue().cast<tensor_network::IndexLabelType>().getSize().getInt());
            }
        }
        for (auto index : rhsIndices) {
            if (!llvm::is_contained(lhsIndices, index)) {
                resultIndices.push_back(index);
                resultShape.push_back(index.cast<TypeAttr>().getValue().cast<tensor_network::IndexLabelType>().getSize().getInt());
            }
        }

        auto resultTensorType = RankedTensorType::get(resultShape, rewriter.getF64Type());
        return tensor_network::TensorWithIndicesType::get(rewriter.getContext(), resultTensorType, rewriter.getArrayAttr(resultIndices));
    }
};

struct SplitSimplification : public OpRewritePattern<tensor_network::ContractMultipleTensorsOp> {
    using OpRewritePattern<tensor_network::ContractMultipleTensorsOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(tensor_network::ContractMultipleTensorsOp op,
                                  PatternRewriter &rewriter) const final {
        return success();
    }
};

struct ContractMultipleTensorsOpSlicing : public OpRewritePattern<tensor_network::ContractMultipleTensorsOp> {
    using OpRewritePattern<tensor_network::ContractMultipleTensorsOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(tensor_network::ContractMultipleTensorsOp op,
                                  PatternRewriter &rewriter) const final {

        if (op->hasAttr("sliced")) {
            return failure();
        }

        tensor_network::IndexLabelType indexToSlice = getIndexToSlice(op);

        int64_t indexSize = indexToSlice.getSize().getInt();
        int64_t numberOfSlices = 2; // TODO: TODO: Decide this dynamically
        std::vector<int64_t> sliceSizes;
        std::vector<int64_t> sliceOffsets;

        // Calculate slice sizes and offsets
        int64_t baseSliceSize = indexSize / numberOfSlices;
        int64_t remainder = indexSize % numberOfSlices;
        int64_t currentOffset = 0;
        for (int i = 0; i < numberOfSlices; ++i) {
            int64_t sliceSize = baseSliceSize + (i < remainder ? 1 : 0);
            sliceSizes.push_back(sliceSize);
            sliceOffsets.push_back(currentOffset);
            currentOffset += sliceSize;
            
            // llvm::errs() << "Slice " << i << ": size = " << sliceSize << ", offset = " << sliceOffsets[i] << "\n";
        }

        std::vector<mlir::Value> tensorsToSlice;
        for (auto tensor : op.getTensors()) {
            auto tensorType = tensor.getType().cast<tensor_network::TensorWithIndicesType>();
            auto tensorIndices = tensorType.getIndices();
            for (auto index : tensorIndices) {
                if (index.cast<TypeAttr>().getValue().cast<tensor_network::IndexLabelType>() == indexToSlice) {
                    tensorsToSlice.push_back(tensor);
                    // llvm::errs() << "Tensor to slice: " << tensor << "\n";
                    break;
                }
            }
        }

        for (auto& tensor : tensorsToSlice) {
            if (auto tensorOp = tensor.getDefiningOp<tensor_network::TensorOp>()) {
                auto valuesAttr = tensorOp.getValue();
                auto tensorType = tensorOp.getType().cast<tensor_network::TensorWithIndicesType>().getTensorType();
                auto constOp = rewriter.create<arith::ConstantOp>(tensorOp.getLoc(), tensorType, valuesAttr);
                
                auto tensorFromValueOp = rewriter.create<tensor_network::TensorFromValueOp>(
                    tensorOp.getLoc(), tensorOp.getType(), constOp);
                
                rewriter.replaceOp(tensorOp, tensorFromValueOp.getResult());
                tensor = tensorFromValueOp.getResult();
            }
        }

        // for (auto tensor : tensorsToSlice) {
        //     llvm::errs() << "Tensor to slice after lowering: " << tensor << "\n";
        //     llvm::errs() << "Tensor argument: " << tensor.getDefiningOp()->getOperand(0) << "\n";
        // }

        bool isSharedIndex = tensorsToSlice.size() > 1;

        std::vector<tensor_network::IndexLabelType> newIndices;

        for (int i = 0; i < numberOfSlices; i++) {
            auto newIndexNameAttr = rewriter.getStringAttr(indexToSlice.getName().str() + "_slice_" + std::to_string(i));
            auto sizeAttr = rewriter.getI64IntegerAttr(sliceSizes[i]);

            auto newIndexLabelType = tensor_network::IndexLabelType::get(rewriter.getContext(), sizeAttr, newIndexNameAttr);
            newIndices.push_back(newIndexLabelType);
        }

        std::vector<std::vector<mlir::Value>> slicedTensors(numberOfSlices);
        for (auto tensor : tensorsToSlice) {
            auto tensorType = tensor.getType().cast<tensor_network::TensorWithIndicesType>();
            auto tensorIndices = tensorType.getIndices();

            size_t slicePos = 0;
            for (; slicePos < tensorIndices.size(); ++slicePos) {
                if (tensorIndices[slicePos].cast<TypeAttr>().getValue().cast<tensor_network::IndexLabelType>() == indexToSlice) {
                    break;
                }
            }

            for (int i = 0; i < numberOfSlices; i++) {
                std::vector<mlir::Attribute> newIndicesAttr;
                SmallVector<int64_t, 4> newShape;
                for (size_t j = 0; j < tensorIndices.size(); ++j) {
                    auto indexLabelType = tensorIndices[j].cast<TypeAttr>().getValue().cast<tensor_network::IndexLabelType>();
                    if (indexLabelType == indexToSlice) {
                        newIndicesAttr.push_back(mlir::TypeAttr::get(newIndices[i]));
                        newShape.push_back(newIndices[i].getSize().getInt());
                    } else {
                        newIndicesAttr.push_back(tensorIndices[j]);
                        newShape.push_back(indexLabelType.getSize().getInt());
                    }
                }

                auto elementType = tensorType.getTensorType().cast<ShapedType>().getElementType();
                auto newTensorType = RankedTensorType::get(newShape, elementType);
                auto newTensorWithIndicesType = tensor_network::TensorWithIndicesType::get(
                    rewriter.getContext(), 
                    newTensorType, 
                    rewriter.getArrayAttr(newIndicesAttr));

                SmallVector<OpFoldResult, 4> offsets(tensorIndices.size(), rewriter.getIndexAttr(0));
                SmallVector<OpFoldResult, 4> sizes;
                for (size_t j = 0; j < tensorIndices.size(); ++j) {
                    if (j == slicePos) {
                        offsets[j] = rewriter.getIndexAttr(sliceOffsets[i]);
                        sizes.push_back(rewriter.getIndexAttr(sliceSizes[i]));
                    } else {
                        sizes.push_back(rewriter.getIndexAttr(
                            tensorIndices[j].cast<TypeAttr>().getValue().cast<tensor_network::IndexLabelType>().getSize().getInt()));
                    }
                }

                auto slicedTensorValue = rewriter.create<tensor::ExtractSliceOp>(
                    op.getLoc(), 
                    newTensorType,
                    tensor.getDefiningOp()->getOperand(0), 
                    offsets, 
                    sizes, 
                    SmallVector<OpFoldResult, 4>(tensorIndices.size(), rewriter.getIndexAttr(1)));

                // llvm::errs() << "Sliced tensor " << i << ": " << slicedTensorValue << "\n";

                auto slicedTensor = rewriter.create<tensor_network::TensorFromValueOp>(
                    op.getLoc(), newTensorWithIndicesType, slicedTensorValue);

                slicedTensors[i].push_back(slicedTensor);
            }
        }




        std::vector<Operation*> newOps;

        for (int i = 0; i < numberOfSlices; i++) {
            SmallVector<Value, 4> tensorsForSlice;
            for (const auto& slicedTensor : slicedTensors[i]) {
                tensorsForSlice.push_back(slicedTensor);
            }

            for (auto tensor : op.getTensors()) {
                if (std::find(tensorsToSlice.begin(), tensorsToSlice.end(), tensor) == tensorsToSlice.end()) {
                    tensorsForSlice.push_back(tensor);
                }
            }

            auto newOp = rewriter.create<tensor_network::ContractMultipleTensorsOp>(
                op.getLoc(),
                op.getType(),
                tensorsForSlice
            );

            newOp->setAttr("sliced", rewriter.getUnitAttr());

            newOps.push_back(newOp);
        }

        if (isSharedIndex) {
            Value summedResult = sumResults(rewriter, op, newOps, indexToSlice);
            rewriter.replaceOp(op, summedResult);
        } else {
            return failure();
        }

        return success();
    }

private:
    Value extractTensorValue(PatternRewriter &rewriter, Value v) const {
        if (auto tensorFromValueOp = v.getDefiningOp<tensor_network::TensorFromValueOp>()) {
            return tensorFromValueOp.getOperand();
        }

        llvm::errs() << "Invalid tensor value\n";
        return nullptr;
    }

    Value sumResults(PatternRewriter &rewriter, tensor_network::ContractMultipleTensorsOp op, 
                     const std::vector<Operation*> &newOps,
                     tensor_network::IndexLabelType slicedIndex) const {
        auto loc = op.getLoc();
        Value sumResult = newOps[0]->getResult(0);
        auto resultType = sumResult.getType().cast<tensor_network::TensorWithIndicesType>();

        for (size_t i = 1; i < newOps.size(); ++i) {
            sumResult = rewriter.create<tensor_network::AddOp>(
                loc,
                resultType,
                sumResult,
                newOps[i]->getResult(0)
            );
        }

        return sumResult;
    }

    SmallVector<utils::IteratorType, 4> getNParallelLoopsAttrs(unsigned nLoops) const {
        return SmallVector<utils::IteratorType, 4>(nLoops, utils::IteratorType::parallel);
    }

    tensor_network::IndexLabelType getIndexToSlice(tensor_network::ContractMultipleTensorsOp op) const {
        SmallVector<tensor_network::IndexLabelType, 4> indices;
        for (auto tensor : op.getTensors()) {
            auto tensorType = tensor.getType().cast<tensor_network::TensorWithIndicesType>();
            auto tensorIndices = tensorType.getIndices();
            for (auto index : tensorIndices) {
                indices.push_back(index.cast<TypeAttr>().getValue().cast<tensor_network::IndexLabelType>());
            }
        }

        tensor_network::IndexLabelType indexToSlice;
        int64_t maxSize = 0;
        for (auto index : indices) {
            if (index.getSize().getInt() > maxSize) {
                maxSize = index.getSize().getInt();
                indexToSlice = index;
            }
        }

        // llvm::errs() << "Index To Slice: " << indexToSlice.getName() << "\n";

        return indexToSlice;
    }
};


struct ContractMultipleTensorsOpSlicingMultithreading : public OpRewritePattern<tensor_network::ContractMultipleTensorsOp> {
    using OpRewritePattern<tensor_network::ContractMultipleTensorsOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(tensor_network::ContractMultipleTensorsOp op,
                                  PatternRewriter &rewriter) const override {

        if (op->hasAttr("sliced")) {
            return failure();
        }

        tensor_network::IndexLabelType indexToSlice = getIndexToSlice(op);

        int64_t indexSize = indexToSlice.getSize().getInt();
        int64_t numberOfSlices = 2; // TODO: Test for other slices and determine best number
        std::vector<int64_t> sliceSizes;
        std::vector<int64_t> sliceOffsets;

        // Calculate slice sizes and offsets
        int64_t baseSliceSize = indexSize / numberOfSlices;
        int64_t remainder = indexSize % numberOfSlices;
        int64_t currentOffset = 0;
        for (int i = 0; i < numberOfSlices; ++i) {
            int64_t sliceSize = baseSliceSize + (i < remainder ? 1 : 0);
            sliceSizes.push_back(sliceSize);
            sliceOffsets.push_back(currentOffset);
            currentOffset += sliceSize;
            
            // llvm::errs() << "Slice " << i << ": size = " << sliceSize << ", offset = " << sliceOffsets[i] << "\n";
        }

        std::vector<mlir::Value> tensorsToSlice;

        for (auto tensor : op.getTensors()) {
            auto tensorType = tensor.getType().cast<tensor_network::TensorWithIndicesType>();
            auto tensorIndices = tensorType.getIndices();
            for (auto index : tensorIndices) {
                if (index.cast<TypeAttr>().getValue().cast<tensor_network::IndexLabelType>() == indexToSlice) {
                    tensorsToSlice.push_back(tensor);
                    // llvm::errs() << "Tensor to slice: " << tensor << "\n";
                    break;
                }
            }
        }

        for (auto& tensor : tensorsToSlice) {
            if (auto tensorOp = tensor.getDefiningOp<tensor_network::TensorOp>()) {
                auto valuesAttr = tensorOp.getValue();
                auto tensorType = tensorOp.getType().cast<tensor_network::TensorWithIndicesType>().getTensorType();
                auto constOp = rewriter.create<arith::ConstantOp>(tensorOp.getLoc(), tensorType, valuesAttr);
                
                auto tensorFromValueOp = rewriter.create<tensor_network::TensorFromValueOp>(
                    tensorOp.getLoc(), tensorOp.getType(), constOp);
                
                rewriter.replaceOp(tensorOp, tensorFromValueOp.getResult());
                tensor = tensorFromValueOp.getResult();
            }
        }

        bool isSharedIndex = tensorsToSlice.size() > 1;

        std::vector<tensor_network::IndexLabelType> newIndices;

        for (int i = 0; i < numberOfSlices; i++) {
            auto newIndexNameAttr = rewriter.getStringAttr(indexToSlice.getName().str() + "_slice_" + std::to_string(i));
            auto sizeAttr = rewriter.getI64IntegerAttr(sliceSizes[i]);

            auto newIndexLabelType = tensor_network::IndexLabelType::get(rewriter.getContext(), sizeAttr, newIndexNameAttr);
            newIndices.push_back(newIndexLabelType);
        }

        std::vector<std::vector<mlir::Value>> slicedTensors(numberOfSlices);
        for (auto tensor : tensorsToSlice) {
            auto tensorType = tensor.getType().cast<tensor_network::TensorWithIndicesType>();
            auto tensorIndices = tensorType.getIndices();

            size_t slicePos = 0;
            for (; slicePos < tensorIndices.size(); ++slicePos) {
                if (tensorIndices[slicePos].cast<TypeAttr>().getValue().cast<tensor_network::IndexLabelType>() == indexToSlice) {
                    break;
                }
            }

            for (int i = 0; i < numberOfSlices; i++) {
                std::vector<mlir::Attribute> newIndicesAttr;
                SmallVector<int64_t, 4> newShape;
                for (size_t j = 0; j < tensorIndices.size(); ++j) {
                    auto indexLabelType = tensorIndices[j].cast<TypeAttr>().getValue().cast<tensor_network::IndexLabelType>();
                    if (indexLabelType == indexToSlice) {
                        newIndicesAttr.push_back(mlir::TypeAttr::get(newIndices[i]));
                        newShape.push_back(newIndices[i].getSize().getInt());
                    } else {
                        newIndicesAttr.push_back(tensorIndices[j]);
                        newShape.push_back(indexLabelType.getSize().getInt());
                    }
                }

                auto elementType = tensorType.getTensorType().cast<ShapedType>().getElementType();
                auto newTensorType = RankedTensorType::get(newShape, elementType);
                auto newTensorWithIndicesType = tensor_network::TensorWithIndicesType::get(
                    rewriter.getContext(), 
                    newTensorType, 
                    rewriter.getArrayAttr(newIndicesAttr));

                SmallVector<OpFoldResult, 4> offsets(tensorIndices.size(), rewriter.getIndexAttr(0));
                SmallVector<OpFoldResult, 4> sizes;
                for (size_t j = 0; j < tensorIndices.size(); ++j) {
                    if (j == slicePos) {
                        offsets[j] = rewriter.getIndexAttr(sliceOffsets[i]);
                        sizes.push_back(rewriter.getIndexAttr(sliceSizes[i]));
                    } else {
                        sizes.push_back(rewriter.getIndexAttr(
                            tensorIndices[j].cast<TypeAttr>().getValue().cast<tensor_network::IndexLabelType>().getSize().getInt()));
                    }
                }

                auto slicedTensorValue = rewriter.create<tensor::ExtractSliceOp>(
                    op.getLoc(), 
                    newTensorType,
                    tensor.getDefiningOp()->getOperand(0), 
                    offsets, 
                    sizes, 
                    SmallVector<OpFoldResult, 4>(tensorIndices.size(), rewriter.getIndexAttr(1)));

                // llvm::errs() << "Sliced tensor " << i << ": " << slicedTensorValue << "\n";

                auto slicedTensor = rewriter.create<tensor_network::TensorFromValueOp>(
                    op.getLoc(), newTensorWithIndicesType, slicedTensorValue);

                slicedTensors[i].push_back(slicedTensor);
            }
        }




        std::vector<Value> asyncResults;
        std::vector<Value> asyncTokens;

        for (int i = 0; i < numberOfSlices; i++) {
            SmallVector<Value, 4> tensorsForSlice;
            for (const auto& slicedTensor : slicedTensors[i]) {
                tensorsForSlice.push_back(slicedTensor);
            }

            for (auto tensor : op.getTensors()) {
                if (std::find(tensorsToSlice.begin(), tensorsToSlice.end(), tensor) == tensorsToSlice.end()) {
                    tensorsForSlice.push_back(tensor);
                }
            }

            // Create async.execute operation
            auto asyncOp = rewriter.create<async::ExecuteOp>(
                op.getLoc(),
                TypeRange{op.getType().getTensorType()},
                ValueRange{},
                ValueRange{},
                [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange) {
                    auto newOp = nestedBuilder.create<tensor_network::ContractMultipleTensorsOp>(
                        nestedLoc,
                        op.getType(),
                        tensorsForSlice
                    );
                    newOp->setAttr("sliced", nestedBuilder.getUnitAttr());

                    nestedBuilder.create<async::YieldOp>(nestedLoc, newOp.getResult());
                }
            );

            asyncResults.push_back(asyncOp.getResult(1));  // The actual result value
            asyncTokens.push_back(asyncOp.getResult(0));   // The async token
        }

        if (isSharedIndex) {

            // Now sum the results
            SmallVector<Value, 4> awaitedResults;
            for (auto asyncResult : asyncResults) {
                auto awaitedValue = rewriter.create<async::AwaitOp>(
                    op.getLoc(), 
                    asyncResult
                );
                awaitedResults.push_back(awaitedValue.getResult());
            }

            Value summedResult = sumResults(rewriter, op, awaitedResults, indexToSlice);
            rewriter.replaceOp(op, summedResult);
        } else {
            return failure();
        }

        return success();
    }

private:
    Value extractTensorValue(PatternRewriter &rewriter, Value v) const {
        if (auto tensorFromValueOp = v.getDefiningOp<tensor_network::TensorFromValueOp>()) {
            return tensorFromValueOp.getOperand();
        }

        llvm::errs() << "Invalid tensor value\n";
        return nullptr;
    }

    Value sumResults(PatternRewriter &rewriter, tensor_network::ContractMultipleTensorsOp op, 
                     const SmallVector<Value, 4> &results,
                     tensor_network::IndexLabelType slicedIndex) const {
        auto loc = op.getLoc();
        Value sumResult = results[0];

        // llvm::errs() << "Summing results\n";

        auto rank = op.getType().getTensorType().cast<RankedTensorType>().getRank();
        auto zeroConstant = rewriter.create<arith::ConstantOp>(
            loc, 
            op.getType().getTensorType(), 
            rewriter.getZeroAttr(op.getType().getTensorType())
        );

        for (size_t i = 1; i < results.size(); ++i) {
            sumResult = rewriter.create<linalg::GenericOp>(
                op.getLoc(),
                TypeRange{op.getType().getTensorType()},
                ValueRange{sumResult, results[i]},
                ValueRange{zeroConstant},
                ArrayRef<AffineMap>{
                    rewriter.getMultiDimIdentityMap(rank),
                    rewriter.getMultiDimIdentityMap(rank),
                    rewriter.getMultiDimIdentityMap(rank)
                },
                getNParallelLoopsAttrs(rank),
                [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
                    Value add = nestedBuilder.create<arith::AddFOp>(nestedLoc, args[0], args[1]);
                    nestedBuilder.create<linalg::YieldOp>(nestedLoc, add);
                }
            ).getResult(0);
        }

        return sumResult;
    }

    SmallVector<utils::IteratorType, 4> getNParallelLoopsAttrs(unsigned nLoops) const {
        return SmallVector<utils::IteratorType, 4>(nLoops, utils::IteratorType::parallel);
    }

    tensor_network::IndexLabelType getIndexToSlice(tensor_network::ContractMultipleTensorsOp op) const {
        SmallVector<tensor_network::IndexLabelType, 4> indices;
        for (auto tensor : op.getTensors()) {
            auto tensorType = tensor.getType().cast<tensor_network::TensorWithIndicesType>();
            auto tensorIndices = tensorType.getIndices();
            for (auto index : tensorIndices) {
                indices.push_back(index.cast<TypeAttr>().getValue().cast<tensor_network::IndexLabelType>());
            }
        }

        tensor_network::IndexLabelType indexToSlice;
        int64_t maxSize = 0;
        for (auto index : indices) {
            if (index.getSize().getInt() > maxSize) {
                maxSize = index.getSize().getInt();
                indexToSlice = index;
            }
        }

        // llvm::errs() << "Index To Slice: " << indexToSlice.getName() << "\n";

        return indexToSlice;
    }
};

struct ContractMultipleTensorsOpGreedyGrayKourtis : public OpRewritePattern<tensor_network::ContractMultipleTensorsOp> {
    ContractMultipleTensorsOpGreedyGrayKourtis(MLIRContext *context, double alpha = 1.0)
        : OpRewritePattern<tensor_network::ContractMultipleTensorsOp>(context), alpha(alpha) {}

    LogicalResult matchAndRewrite(tensor_network::ContractMultipleTensorsOp op,
                                  PatternRewriter &rewriter) const override {
        SmallVector<Value, 4> tensors(op.getTensors().begin(), op.getTensors().end());

        while (tensors.size() > 1) {
            int bestI = 0, bestJ = 1;
            double bestCost = std::numeric_limits<double>::max();

            for (size_t i = 0; i < tensors.size(); ++i) {
                for (size_t j = i + 1; j < tensors.size(); ++j) {
                    double cost = calculateContractionCost(tensors[i], tensors[j], rewriter);
                    if (cost < bestCost) {
                        bestCost = cost;
                        bestI = i;
                        bestJ = j;
                    }
                }
            }

            auto resultType = determineContractedTensorType(tensors[bestI], tensors[bestJ], rewriter);
            auto contractOp = rewriter.create<tensor_network::ContractTensorsOp>(
                op.getLoc(), 
                resultType,
                tensors[bestI], tensors[bestJ]);

            tensors[bestI] = contractOp.getResult();
            tensors.erase(tensors.begin() + bestJ);
        }

        rewriter.replaceOp(op, tensors[0]);
        return success();
    }

private:
    double alpha;

    int64_t calculateTensorSize(const tensor_network::TensorWithIndicesType& tensorType) const {
        auto indices = tensorType.getIndices();
        int64_t size = 1;
        for (auto index : indices) {
            size *= index.cast<TypeAttr>().getValue().cast<tensor_network::IndexLabelType>().getSize().getInt();
        }
        return size;
    }

    double calculateContractionCost(Value lhs, Value rhs, PatternRewriter &rewriter) const {
        auto lhsType = lhs.getType().cast<tensor_network::TensorWithIndicesType>();
        auto rhsType = rhs.getType().cast<tensor_network::TensorWithIndicesType>();
        
        int64_t sizeT_i = calculateTensorSize(lhsType);
        int64_t sizeT_j = calculateTensorSize(rhsType);
        
        auto resultType = determineContractedTensorType(lhs, rhs, rewriter);
        int64_t sizeT_k = calculateTensorSize(resultType);

        return static_cast<double>(sizeT_k) - alpha * (sizeT_i + sizeT_j);
    }

    tensor_network::TensorWithIndicesType determineContractedTensorType(Value lhs, Value rhs, PatternRewriter &rewriter) const {
        auto lhsType = lhs.getType().cast<tensor_network::TensorWithIndicesType>();
        auto rhsType = rhs.getType().cast<tensor_network::TensorWithIndicesType>();
        auto lhsIndices = lhsType.getIndices();
        auto rhsIndices = rhsType.getIndices();

        SmallVector<mlir::Attribute> resultIndices;
        SmallVector<int64_t> resultShape;

        for (auto index : lhsIndices) {
            if (std::find(rhsIndices.begin(), rhsIndices.end(), index) == rhsIndices.end()) {
                resultIndices.push_back(index);
                resultShape.push_back(index.cast<TypeAttr>().getValue().cast<tensor_network::IndexLabelType>().getSize().getInt());
            }
        }
        for (auto index : rhsIndices) {
            if (std::find(lhsIndices.begin(), lhsIndices.end(), index) == lhsIndices.end()) {
                resultIndices.push_back(index);
                resultShape.push_back(index.cast<TypeAttr>().getValue().cast<tensor_network::IndexLabelType>().getSize().getInt());
            }
        }

        auto resultTensorType = RankedTensorType::get(resultShape, rewriter.getF64Type());
        return tensor_network::TensorWithIndicesType::get(rewriter.getContext(), resultTensorType, rewriter.getArrayAttr(resultIndices));
    }
};

struct ContractMultipleTensorsOpGreedy : public OpRewritePattern<tensor_network::ContractMultipleTensorsOp> {
    using OpRewritePattern<tensor_network::ContractMultipleTensorsOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(tensor_network::ContractMultipleTensorsOp op,
                                  PatternRewriter &rewriter) const override {

        // Get the list of tensors to contract
        SmallVector<Value, 4> tensors(op.getTensors().begin(), op.getTensors().end());

        // Greedy algorithm to determine contraction order
        while (tensors.size() > 1) {
            int bestI = 0, bestJ = 1;
            int64_t bestCost = std::numeric_limits<int64_t>::max();

            for (size_t i = 0; i < tensors.size(); ++i) {
                for (size_t j = i + 1; j < tensors.size(); ++j) {
                    int64_t cost = calculateContractionCost(tensors[i], tensors[j]);
                    if (cost < bestCost) {
                        bestCost = cost;
                        bestI = i;
                        bestJ = j;
                    }
                }
            }

            // Contract the best pair
            auto resultType = determineContractedTensorType(tensors[bestI], tensors[bestJ], rewriter);
            auto contractOp = rewriter.create<tensor_network::ContractTensorsOp>(
                op.getLoc(), 
                resultType,
                tensors[bestI], tensors[bestJ]);

            // Replace the contracted tensors with the result
            tensors[bestI] = contractOp.getResult();
            tensors.erase(tensors.begin() + bestJ);
        }

        // The final result is in tensors[0]

        rewriter.replaceOp(op, tensors[0]);

        return success();
    }

private:
    int64_t calculateContractionCost(Value lhs, Value rhs) const {
        auto lhsType = lhs.getType().cast<tensor_network::TensorWithIndicesType>();
        auto rhsType = rhs.getType().cast<tensor_network::TensorWithIndicesType>();
        auto lhsIndices = lhsType.getIndices();
        auto rhsIndices = rhsType.getIndices();

        int64_t cost = 1;
        for (auto index : lhsIndices) {
            if (std::find(rhsIndices.begin(), rhsIndices.end(), index) == rhsIndices.end()) {
                cost *= index.cast<TypeAttr>().getValue().cast<tensor_network::IndexLabelType>().getSize().getInt();
            }
        }
        for (auto index : rhsIndices) {
            if (std::find(lhsIndices.begin(), lhsIndices.end(), index) == lhsIndices.end()) {
                cost *= index.cast<TypeAttr>().getValue().cast<tensor_network::IndexLabelType>().getSize().getInt();
            }
        }
        return cost;
    }

    tensor_network::TensorWithIndicesType determineContractedTensorType(Value lhs, Value rhs, PatternRewriter &rewriter) const {
        auto lhsType = lhs.getType().cast<tensor_network::TensorWithIndicesType>();
        auto rhsType = rhs.getType().cast<tensor_network::TensorWithIndicesType>();
        auto lhsIndices = lhsType.getIndices();
        auto rhsIndices = rhsType.getIndices();

        SmallVector<mlir::Attribute> resultIndices;
        SmallVector<int64_t> resultShape;

        for (auto index : lhsIndices) {
            if (std::find(rhsIndices.begin(), rhsIndices.end(), index) == rhsIndices.end()) {
                resultIndices.push_back(index);
                resultShape.push_back(index.cast<TypeAttr>().getValue().cast<tensor_network::IndexLabelType>().getSize().getInt());
            }
        }
        for (auto index : rhsIndices) {
            if (std::find(lhsIndices.begin(), lhsIndices.end(), index) == lhsIndices.end()) {
                resultIndices.push_back(index);
                resultShape.push_back(index.cast<TypeAttr>().getValue().cast<tensor_network::IndexLabelType>().getSize().getInt());
            }
        }

        auto resultTensorType = RankedTensorType::get(resultShape, rewriter.getF64Type());
        return tensor_network::TensorWithIndicesType::get(rewriter.getContext(), resultTensorType, rewriter.getArrayAttr(resultIndices));
    }
};

struct WriteFinalTensorPattern : public OpRewritePattern<mlir::func::FuncOp> {
    using OpRewritePattern<mlir::func::FuncOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(mlir::func::FuncOp op,
                                  PatternRewriter &rewriter) const override {

        // llvm::errs() << "Matching FuncOp: " << op.getName() << "\n";

        // Check if the operation has already been rewritten
        if (op->hasAttr("rewritten")) {
            return failure();
        }

        // Check if the main function is the one we want to modify
        if (op.getName() != "main") {
            llvm::errs() << "Only modifying the main function\n";
            return failure();
        }

        // Find the last operation that returns a TensorWithIndicesType in the main function
        mlir::Value lastTensorWithIndicesOp = nullptr;
        for (auto &innerOp : llvm::reverse(op.getBody().front())) {
            bool found = false;
            if (innerOp.getNumResults() > 0) {
                for (auto result : innerOp.getResults()) {
                    if (result.getType().isa<tensor_network::TensorWithIndicesType>()) {
                        lastTensorWithIndicesOp = result;
                        found = true;
                        break;
                    }
                }
            }
            if (found) {
                break;
            }
        }

        if (!lastTensorWithIndicesOp) {
            llvm::errs() << "No operation returning TensorWithIndicesType found in the main function\n";
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
        rewriter.replaceOpWithNewOp<mlir::func::ReturnOp>(returnOp, lastTensorWithIndicesOp);

        // Modify the function type to reflect the new return type and no arguments
        auto funcType = rewriter.getFunctionType({}, lastTensorWithIndicesOp.getType().cast<tensor_network::TensorWithIndicesType>().getTensorType());
        op.setType(funcType);

        op->setAttr("rewritten", rewriter.getUnitAttr());

        return mlir::success();
    }
};

class TensorWithIndicesTypeConverter : public TypeConverter {
public:
    TensorWithIndicesTypeConverter() {
        addConversion([](tensor_network::TensorWithIndicesType tensorWithIndicesType) -> RankedTensorType {

            // llvm::errs() << "Converting TensorWithIndicesType to RankedTensorType\n";

            return tensorWithIndicesType.getTensorType().cast<RankedTensorType>();
        });

        addConversion([](tensor_network::IndexLabelType indexLabelType) -> IntegerType {
            return IntegerType::get(indexLabelType.getContext(), 64);
        });

        addConversion([](RankedTensorType rankedTensorType) -> RankedTensorType {
            return rankedTensorType;
        });
        addConversion([](async::ValueType valueType) -> async::ValueType {
            if (auto tensorWithIndicesType = valueType.getValueType().dyn_cast<tensor_network::TensorWithIndicesType>()) {
                return async::ValueType::get(tensorWithIndicesType.getTensorType());
            } else {
                return valueType;
            }
        });
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
            mlir::tensor::TensorDialect, mlir::bufferization::BufferizationDialect,
            mlir::async::AsyncDialect, mlir::BuiltinDialect>();
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
    target.addLegalDialect<mlir::arith::ArithDialect,
        mlir::linalg::LinalgDialect, mlir::tensor::TensorDialect,
        mlir::memref::MemRefDialect, mlir::bufferization::BufferizationDialect,
        mlir::BuiltinDialect>();

    {
        // module.dump();

        RewritePatternSet writeFinalTensorPatterns(&getContext());
        writeFinalTensorPatterns.add<WriteFinalTensorPattern>(&getContext());

        if (failed(applyPatternsAndFoldGreedily(module, std::move(writeFinalTensorPatterns)))) {
            llvm::errs() << "Failed to apply the WriteFinalTensorPattern\n";
            signalPassFailure();
            return;
        }

        // module.dump();
    }

    {
        //Apply rank simplification
        RewritePatternSet rankSimplificationPatterns(&getContext());
        rankSimplificationPatterns.add<RankSimplification>(&getContext());

        // llvm::errs() << "Module before rank simplification:\n";
        // module.dump();

        if (failed(applyPatternsAndFoldGreedily(module, std::move(rankSimplificationPatterns)))) {
            signalPassFailure();
            return;
        }


        // llvm::errs() << "Module after rank simplification:\n";
        // module.dump();
    }

    // {
    //
    //     // llvm::errs() << "Module before lowering:\n";
    //     // module.dump();
    //
    //     //Step -1: Slice the index
    //     RewritePatternSet slicingPatterns(&getContext());
    //     slicingPatterns.add<ContractMultipleTensorsOpSlicing>(&getContext());
    //
    //     if (failed(applyPatternsAndFoldGreedily(module, std::move(slicingPatterns)))) {
    //         signalPassFailure();
    //         return;
    //     }
    //
    //     // llvm::errs() << "Module after slicing\n";
    //     // module.dump();
    // }


    // {
    //     // llvm::errs() << "Step 0: Determine the contraction order of ContractMultiple\n";
    //
    //     RewritePatternSet contractionOrderPatterns(&getContext());
    //     contractionOrderPatterns.add<ContractMultipleTensorsOpGreedy>(&getContext());
    //
    //     if (failed(applyPatternsAndFoldGreedily(module, std::move(contractionOrderPatterns)))) {
    //         signalPassFailure();
    //         return;
    //     }
    //
    //     // module.dump();
    // }

    {
        // llvm::errs() << "Step 0: Determine the contraction order of ContractMultiple\n";

        RewritePatternSet contractionOrderPatterns(&getContext());
        contractionOrderPatterns.add<ContractMultipleTensorsOpGreedyGrayKourtis>(&getContext(), 1.0);

        if (failed(applyPatternsAndFoldGreedily(module, std::move(contractionOrderPatterns)))) {

            signalPassFailure();
            return;
        }

        // module.dump();
    }


    {
        // llvm::DebugFlag = true;

        TensorWithIndicesTypeConverter typeConverter;
        target.addIllegalDialect<tensor_network::TensorNetworkDialect>();



        RewritePatternSet loweringPatterns(context);
        loweringPatterns.add<TensorDeclOpLowering, TensorOpLowering, IndexOpLowering, 
            ContractOpLowering, AddOpLowering, FuncOpLowering, RemoveTensorFromValueOp,
            AsyncYieldOpLowering, AsyncAwaitOpLowering>(typeConverter, context);

        populateReturnOpTypeConversionPattern(loweringPatterns, typeConverter);
        target.addDynamicallyLegalOp<mlir::func::FuncOp>(
            [&](mlir::func::FuncOp op) {
                return llvm::none_of(op->getResultTypes(),
                                     [](Type type) { return type.isa<tensor_network::TensorWithIndicesType>(); });
            }
        );
        target.addDynamicallyLegalOp<func::ReturnOp>(
            [&](func::ReturnOp op) { 
                return llvm::none_of(op->getOperandTypes(),
                                     [](Type type) { return type.isa<tensor_network::TensorWithIndicesType>(); });
            });
        target.addDynamicallyLegalOp<async::ExecuteOp>(
            [&](async::ExecuteOp op) {
                return llvm::none_of(op->getResultTypes(),
                                     [](Type type) { return type.isa<tensor_network::TensorWithIndicesType>(); });
            });
        target.addDynamicallyLegalOp<async::YieldOp>(
            [&](async::YieldOp op) {
                return llvm::none_of(op->getOperandTypes(),
                                     [](Type type) { return type.isa<tensor_network::TensorWithIndicesType>(); });
            });
        target.addDynamicallyLegalOp<async::AwaitOp>(
            [&](async::AwaitOp op) {
                return llvm::none_of(op->getOperandTypes(), 
                                     [](Type type) { 
                                     auto asyncValueType = type.dyn_cast<async::ValueType>();
                                     if (!asyncValueType) { return false; }
                                     return asyncValueType.getValueType().isa<tensor_network::TensorWithIndicesType>();
                                     });
            });

        LogicalResult res = applyFullConversion(module, target, std::move(loweringPatterns));

        if (failed(res)) {
            signalPassFailure();
            return;
        }
    }

    if (failed(mlir::verify(module))) {
        llvm::errs() << "Error in module verification after all transformations\n";
        signalPassFailure();
        return;
    }

    // module.dump();

}


std::unique_ptr<mlir::Pass> mlir::tensor_network::createTensorNetworkNaiveLoweringPass() {
    return std::make_unique<TensorNetworkNaiveLoweringPass>();
}
