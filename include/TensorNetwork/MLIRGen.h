#ifndef TN_MLIRGEN_H
#define TN_MLIRGEN_H

#include <memory>

namespace mlir {
class MLIRContext;
template <typename OpTy>
class OwningOpRef;
class ModuleOp;
}  // namespace mlir

namespace tensor_network {
class ModuleAST;

/// Emit IR for the given Toy moduleAST, returns a newly created MLIR module
/// or nullptr on failure.
mlir::OwningOpRef<mlir::ModuleOp> mlirGen(mlir::MLIRContext &context,
                                          ModuleAST &moduleAST);
}  // namespace tensor_network

#endif  // TN_MLIRGEN_HIEEE/ACM International Symposium on Code Generation and Optimization (CGO)