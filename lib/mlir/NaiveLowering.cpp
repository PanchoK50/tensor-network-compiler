#include "TensorNetwork/Passes.h"
#include "TensorNetwork/TensorNetworkDialect.h"
#include "TensorNetwork/TensorNetworkOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
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

using namespace mlir;

//===----------------------------------------------------------------------===//
// TensorNetworkToNaive RewritePatterns
//===----------------------------------------------------------------------===//

struct NaiveTensorOpLowering : public OpRewritePattern<tensor_network::TensorOp> {
    using OpRewritePattern<tensor_network::TensorOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(tensor_network::TensorOp op,
                                  PatternRewriter &rewriter) const final {
        auto constantOp = rewriter.create<mlir::arith::ConstantOp>(op.getLoc(), op.getValue().getType(), op.getValue());
        rewriter.replaceOp(op, constantOp.getResult());
        return success();
    }
};

struct IndexOpLowering : public OpRewritePattern<tensor_network::IndexOp> {
    using OpRewritePattern<tensor_network::IndexOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(tensor_network::IndexOp op,
                                  PatternRewriter &rewriter) const final {
        auto constantOp = rewriter.create<mlir::arith::ConstantIndexOp>(op.getLoc(), op.getSize());
        rewriter.replaceOp(op, constantOp.getResult());
        return success();
    }
};

struct ContractOpLowering : public OpRewritePattern<tensor_network::ContractTensorsOp> {
    using OpRewritePattern<tensor_network::ContractTensorsOp>::OpRewritePattern;
    LogicalResult matchAndRewrite(tensor_network::ContractTensorsOp op,
                                  PatternRewriter &rewriter) const final {
        Value lhs = op.getLhs();
        Value rhs = op.getRhs();

        auto lhsIndices = lhs.getDefiningOp<tensor_network::TensorOp>().getIndices();
        auto rhsIndices = rhs.getDefiningOp<tensor_network::TensorOp>().getIndices();
        SmallVector<Value> uniqueIndices;
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

        SmallVector<Value> commonIndices;
        for (auto lhsIndex : lhsIndices) {
            if (std::find(rhsIndices.begin(), rhsIndices.end(), lhsIndex) != rhsIndices.end()) {
                commonIndices.push_back(lhsIndex);
            }
        }

        if (commonIndices.empty()) {
            llvm::errs() << "No common indices found. Lowering failed.\n";
            return failure();
        }

        auto lhsType = lhs.getType().cast<RankedTensorType>();
        // auto rhsType = rhs.getType().cast<RankedTensorType>(); //Just use one of them, only to get type of result tensor

        SmallVector<int64_t> resultShape;
        for (auto lhsIndex : lhsIndices) {
            if (std::find(commonIndices.begin(), commonIndices.end(), lhsIndex) == commonIndices.end()) {
                auto indexOp = lhsIndex.getDefiningOp<tensor_network::IndexOp>();
                int64_t size = indexOp.getSize();
                resultShape.push_back(size);
            }
        }
        for (auto rhsIndex : rhsIndices) {
            if (std::find(commonIndices.begin(), commonIndices.end(), rhsIndex) == commonIndices.end()) {
                auto indexOp = rhsIndex.getDefiningOp<tensor_network::IndexOp>();
                int64_t size = indexOp.getSize();
                resultShape.push_back(size);
            }
        }

        auto resultType = RankedTensorType::get(resultShape, lhsType.getElementType());

        auto zeroConstant = rewriter.create<mlir::arith::ConstantOp>(op.getLoc(), resultType, rewriter.getZeroAttr(resultType));

        SmallVector<AffineMap, 3> indexingMaps;

        // Create an affine map for the lhs tensor
        SmallVector<AffineExpr, 4> lhsExprs;
        for (unsigned i = 0; i < uniqueIndices.size(); ++i) {
            if (std::find(lhsIndices.begin(), lhsIndices.end(), uniqueIndices[i]) != lhsIndices.end()) {
                lhsExprs.push_back(rewriter.getAffineDimExpr(i));
            }
        }
        indexingMaps.push_back(AffineMap::get(uniqueIndices.size(), 0, lhsExprs, rewriter.getContext()));

        // Create an affine map for the rhs tensor
        SmallVector<AffineExpr, 4> rhsExprs;
        for (unsigned i = 0; i < uniqueIndices.size(); ++i) {
            if (std::find(rhsIndices.begin(), rhsIndices.end(), uniqueIndices[i]) != rhsIndices.end()) {
                rhsExprs.push_back(rewriter.getAffineDimExpr(i));
            }
        }
        indexingMaps.push_back(AffineMap::get(uniqueIndices.size(), 0, rhsExprs, rewriter.getContext()));

        // Create an affine map for the result tensor
        SmallVector<AffineExpr, 4> resultExprs;
        for (unsigned i = 0; i < uniqueIndices.size(); ++i) {
            if (std::find(commonIndices.begin(), commonIndices.end(), uniqueIndices[i]) == commonIndices.end()) {
                resultExprs.push_back(rewriter.getAffineDimExpr(i));
            }
        }
        indexingMaps.push_back(AffineMap::get(uniqueIndices.size(), 0, resultExprs, rewriter.getContext()));

        // Determine iterator types
        SmallVector<utils::IteratorType> iteratorTypes;

        for (unsigned i = 0; i < uniqueIndices.size(); ++i) {
            if (std::find(commonIndices.begin(), commonIndices.end(), uniqueIndices[i]) != commonIndices.end()) {
                iteratorTypes.push_back(utils::IteratorType::reduction);
            } else {
                iteratorTypes.push_back(utils::IteratorType::parallel);
            }
        }

        auto genericOp = rewriter.create<linalg::GenericOp>(
            op.getLoc(), TypeRange{resultType}, ValueRange{lhs, rhs}, ValueRange{zeroConstant.getResult()},
            indexingMaps, iteratorTypes,
            [](OpBuilder &builder, Location loc, ValueRange args) {
                auto mulOp = builder.create<mlir::arith::MulFOp>(loc, args[0], args[1]);
                auto addOp = builder.create<mlir::arith::AddFOp>(loc, mulOp, args[2]);
                builder.create<linalg::YieldOp>(loc, addOp.getResult());
            });

        // // Print all debug information available about genericOp
        // llvm::errs() << "genericOp: " << genericOp << "\n";
        // llvm::errs() << "genericOp location: " << genericOp.getLoc() << "\n";
        // llvm::errs() << "genericOp result type: " << genericOp.getResultTypes() << "\n";
        // llvm::errs() << "genericOp operands: ";
        // for (auto operand : genericOp.getOperands()) {
        //     llvm::errs() << operand << " ";
        // }
        // llvm::errs() << "\n";
        // llvm::errs() << "genericOp indexing maps: ";
        // for (auto map : genericOp.getIndexingMaps()) {
        //     llvm::errs() << map << " ";
        // }
        // llvm::errs() << "\n";
        // llvm::errs() << "genericOp iterator types: ";
        // for (auto iterType : genericOp.getIteratorTypes()) {
        //     llvm::errs() << iterType << " ";
        // }
        // llvm::errs() << "\n";
        // llvm::errs() << "genericOp body: " << genericOp.getBody() << "\n";
        // llvm::errs() << "genericOp body operations: ";
        // for (auto &op : genericOp.getBody()->getOperations()) {
        //     llvm::errs() << op << " ";
        // }
        // llvm::errs() << "\n";

        rewriter.replaceOp(op, genericOp.getResult(0));

        return success();
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
                        mlir::memref::MemRefDialect, mlir::arith::ArithDialect>();
    }

    void runOnOperation() final;
};
}  // namespace

void TensorNetworkNaiveLoweringPass::runOnOperation() {
    ConversionTarget target(getContext());

    target.addLegalDialect<arith::ArithDialect, linalg::LinalgDialect, func::FuncDialect,
                           memref::MemRefDialect>();

    target.addIllegalDialect<tensor_network::TensorNetworkDialect>();

    RewritePatternSet patterns(&getContext());
    patterns.add<NaiveTensorOpLowering, IndexOpLowering, ContractOpLowering>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<mlir::Pass> mlir::tensor_network::createTensorNetworkNaiveLoweringPass() {
    return std::make_unique<TensorNetworkNaiveLoweringPass>();
}
