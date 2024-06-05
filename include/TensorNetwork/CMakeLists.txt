
add_mlir_dialect(TensorNetworkOps tensor_network)
add_mlir_doc(TensorNetworkDialect TensorNetworkDialect TensorNetwork/ -gen-dialect-doc)
add_mlir_doc(TensorNetworkOps TensorNetworkOps TensorNetwork/ -gen-op-doc)

set(LLVM_TARGET_DEFINITIONS TensorNetworkPasses.td)
mlir_tablegen(TensorNetworkPasses.h.inc -gen-pass-decls -name TensorNetwork)
add_public_tablegen_target(MLIRTensorNetworkPassesIncGen)

