#include <memory>
#include <utility>

#include "TensorNetwork/Passes.h"
#include "TensorNetwork/TensorNetworkDialect.h"
#include "TensorNetwork/TensorNetworkOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "mlir/Conversion/TensorToLinalg/TensorToLinalg.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/LinalgToStandard/LinalgToStandard.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Bufferization/Pipelines/Passes.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
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

namespace {
struct TensorNetworkToLLVMLoweringPass
    : public PassWrapper<TensorNetworkToLLVMLoweringPass, OperationPass<ModuleOp>> {
    MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TensorNetworkToLLVMLoweringPass)

    void getDependentDialects(DialectRegistry &registry) const override {
        registry.insert<LLVM::LLVMDialect, scf::SCFDialect, arith::ArithDialect,
                        memref::MemRefDialect, linalg::LinalgDialect,
                        func::FuncDialect, affine::AffineDialect>();
    }
    void runOnOperation() final;
};
}  // namespace

void TensorNetworkToLLVMLoweringPass::runOnOperation() {
    LLVMConversionTarget target(getContext());

    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addLegalOp<mlir::ModuleOp>();

    target.addIllegalDialect<arith::ArithDialect>();
    target.addIllegalDialect<linalg::LinalgDialect>();
    target.addIllegalDialect<affine::AffineDialect>();
    target.addIllegalDialect<scf::SCFDialect>();
    target.addIllegalDialect<func::FuncDialect>();
    target.addIllegalDialect<memref::MemRefDialect>();

    LLVMTypeConverter typeConverter(&getContext());

    RewritePatternSet patterns(&getContext());

    // target.addIllegalDialect<tensor_network::TensorNetworkDialect, arith::ArithDialect, linalg::LinalgDialect>();

    mlir::arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    mlir::populateTensorToLinalgPatterns(patterns);
    populateAffineToStdConversionPatterns(patterns);
    populateSCFToControlFlowConversionPatterns(patterns);
    populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
    cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);
    mlir::linalg::populateLinalgToStandardConversionPatterns(patterns);
    populateFuncToLLVMConversionPatterns(typeConverter, patterns);

    // auto module = getOperation();
    // if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    //     signalPassFailure();
    // }
    // module.print(llvm::outs());
    // llvm::outs() << "\n";
    mlir::ModuleOp module = getOperation();
    if (mlir::failed(mlir::applyPartialConversion(module, target, std::move(patterns))))
        signalPassFailure();
    // module.print(llvm::outs());
    // llvm::outs() << "\n";
}

/// Create a pass for lowering operations the remaining `TensorNetwork` operations, as
/// well as `Affine` and `Std`, to the LLVM dialect for codegen.
std::unique_ptr<mlir::Pass> mlir::tensor_network::createTensorNetworkToLLVMLoweringPass() {
    return std::make_unique<TensorNetworkToLLVMLoweringPass>();
}
