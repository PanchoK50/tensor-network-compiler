#include "TensorNetwork/Passes.h"
#include "TensorNetwork/TensorNetworkDialect.h"
#include "TensorNetwork/TensorNetworkOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/LinalgTransformOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/IR/Attributes.h"
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

// From toy tutorial
//  Convert the given RankedTensorType into the corresponding MemRefType.
static MemRefType convertTensorToMemRef(RankedTensorType type) {
    return MemRefType::get(type.getShape(), type.getElementType());
}

// From Toy Tutorial
static Value insertAllocAndDealloc(MemRefType type, Location loc,
                                   PatternRewriter &rewriter) {
    auto alloc = rewriter.create<memref::AllocOp>(loc, type);

    // Make sure to allocate at the beginning of the block.
    auto *parentBlock = alloc->getBlock();
    alloc->moveBefore(&parentBlock->front());

    // Make sure to deallocate this alloc at the end of the block. This is fine
    // as toy functions have no control flow.
    auto dealloc = rewriter.create<memref::DeallocOp>(loc, alloc);
    dealloc->moveBefore(&parentBlock->back());
    return alloc;
}
//===----------
// TensorNetworkToLinalg RewritePatterns
//===----------

// Lowering for TensorOp
// Lowering into Affine
struct TensorOpLowering : public ConversionPattern {
    TensorOpLowering(MLIRContext *ctx)
        : ConversionPattern(tensor_network::TensorOp::getOperationName(), 1, ctx) {}

    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const final {
        auto loc = op->getLoc();
        auto tensorOp = cast<tensor_network::TensorOp>(op);
        return success();
    }
};

// Lowering for ContractionEdgeOp
struct ContractionEdgeOpLowering : public ConversionPattern {
    ContractionEdgeOpLowering(MLIRContext *ctx)
        : ConversionPattern(tensor_network::ContractionEdgeOp::getOperationName(), 3, ctx) {}

    LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                                  ConversionPatternRewriter &rewriter) const final {
        auto loc = op->getLoc();
        auto tensorOp = cast<tensor_network::ContractionEdgeOp>(op);
        return success();

        /*
            If using linalg.generic
            For the iterator_types use SmallVector<StringRef, int> iteratorTypesSum{"parallel", "parallel", "reduction", "reduction"}

            indexing_maps: affine_map
            affine_map<(d0, d1, d2, d3) -> (d1+1, 2*d2, d3)>
            for d0 := ...
                for d1 := ...
                    for d2 := ...
                        for d3 := ...
                            a[d1+1][2*d2][d3]

            In order to define an affineMap in C++:
            // indexing_maps = [
                affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
                affine_map<(d0, d1, d2, d3) -> (d0, d1)>
            ]

            This is done by:
                SmallVector<AffineExpr, 2> ncExprs;
                ncExprs.push_back(mlir::getAffineDimExpr(0, context));
                ncExprs.push_back(mlir::getAffineDimExpr(1, context));
                auto ncIndexingMap = AffineMap::get(dimCount = 4, symbolCount = 0, ncExprs, context);
                SmallVector<AffineMap, 2> indexingMaps = {
                    rewriter.getMultiDimIdentityMap(4) // input
                    ncIndexingMap // output
                }
            
            !!! Needs to make sure that the tensor shapes agree with each other, could lead to OutOfBounds
        */


    }
};

namespace {
struct TensorNetworkToLinalgLoweringPass
    : public PassWrapper<TensorNetworkToLinalgLoweringPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(
        TensorNetworkToLinalgLoweringPass)

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<linalg::LinalgDialect, func::FuncDialect,
                        memref::MemRefDialect>();
    }
    void runOnOperation() final;
};

}  // namespace

// Adapted from the toy dialect
void TensorNetworkToLinalgLoweringPass::runOnOperation() {
    ConversionTarget target(getContext());

    target.addLegalDialect<linalg::LinalgDialect, func::FuncDialect, arith::ArithDialect,
                           memref::MemRefDialect, tensor_network::TensorNetworkDialect>();

    // In order to make it fail if any tensor_network Operations are not lowered
    // target.addIllegalDialect<tensor_network::TensorNetworkDialect>();

    // If some operation should still be legal: addDynamicallyLegalOp<Op>
    /*
    target.addDynamicallyLegalOp<toy::PrintOp>([](toy::PrintOp op) {
        return llvm::none_of(op->getOperandTypes(),
                         [](Type type) { return llvm::isa<TensorType>(type); });
    });
     */

    RewritePatternSet patterns(&getContext());
    patterns.add<TensorOpLowering, ContractionEdgeOpLowering>(&getContext());

    if (failed(applyPartialConversion(getOperation(), target, std::move(patterns)))) {
        signalPassFailure();
    }
}

std::unique_ptr<mlir::Pass> mlir::tensor_network::createTensorNetworkToLinalgLoweringPass() {
    return std::make_unique<TensorNetworkToLinalgLoweringPass>();
}
