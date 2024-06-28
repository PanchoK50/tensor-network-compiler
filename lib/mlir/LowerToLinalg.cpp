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


struct TensorOpLowering: public OpRewritePattern<tensor_network::TensorOp> {
    using OpRewritePattern<tensor_network::TensorOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(tensor_network::TensorOp op,
                                PatternRewriter &rewriter) const final {
    
    llvm::errs() << "Lowering a TensorOp\n";


    

    DenseElementsAttr constantValue = op.getValue();
    Location loc = op.getLoc();

    // When lowering the constant operation, we allocate and assign the constant
    // values to a corresponding memref allocation.
    auto tensorType = llvm::cast<RankedTensorType>(op.getType());
    auto memRefType = convertTensorToMemRef(tensorType);
    auto alloc = insertAllocAndDealloc(memRefType, loc, rewriter);

    // We will be generating constant indices up-to the largest dimension.
    // Create these constants up-front to avoid large amounts of redundant
    // operations.
    auto valueShape = memRefType.getShape();
    SmallVector<Value, 8> constantIndices;

    if (!valueShape.empty()) {
      for (auto i : llvm::seq<int64_t>(0, *llvm::max_element(valueShape)))
        constantIndices.push_back(
            rewriter.create<arith::ConstantIndexOp>(loc, i));
    } else {
      // This is the case of a tensor of rank 0.
      constantIndices.push_back(
          rewriter.create<arith::ConstantIndexOp>(loc, 0));
    }

    // The constant operation represents a multi-dimensional constant, so we
    // will need to generate a store for each of the elements. The following
    // functor recursively walks the dimensions of the constant shape,
    // generating a store when the recursion hits the base case.
    SmallVector<Value, 2> indices;
    auto valueIt = constantValue.value_begin<FloatAttr>();
    std::function<void(uint64_t)> storeElements = [&](uint64_t dimension) {
      // The last dimension is the base case of the recursion, at this point
      // we store the element at the given index.
      if (dimension == valueShape.size()) {
        rewriter.create<affine::AffineStoreOp>(
            loc, rewriter.create<arith::ConstantOp>(loc, *valueIt++), alloc,
            llvm::ArrayRef(indices));
        return;
      }

      // Otherwise, iterate over the current dimension and add the indices to
      // the list.
      for (uint64_t i = 0, e = valueShape[dimension]; i != e; ++i) {
        indices.push_back(constantIndices[i]);
        storeElements(dimension + 1);
        indices.pop_back();
      }
    };

    // Start the element storing recursion from the first dimension.
    storeElements(/*dimension=*/0);

    // Replace this operation with the generated alloc.
    rewriter.replaceOp(op, alloc);
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
        auto contractionOp = cast<tensor_network::ContractionEdgeOp>(op);
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

        // Determine if the indices over which we want to contract match the dimensions
        // Right now the tensor contractions only contracts over 2 tensors and 1 index, possibly later more indices
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
    target.addIllegalDialect<tensor_network::TensorNetworkDialect>();

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
