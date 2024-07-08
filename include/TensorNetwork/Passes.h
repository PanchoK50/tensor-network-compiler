#ifndef TENSOR_NETWORK_PASSES_H
#define TENSOR_NETWORK_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
    namespace tensor_network {

        #define GEN_PASS_DECL
        #include "TensorNetwork/TensorNetworkPasses.h.inc"
        
        std::unique_ptr<mlir::Pass> createTensorNetworkToLinalgLoweringPass();
        std::unique_ptr<mlir::Pass> createTensorNetworkNaiveLoweringPass();
        std::unique_ptr<mlir::Pass> createTensorNetworkToLLVMLoweringPass();

    } // namespace tensor_network
} // namespace mlir

#endif // TENSOR_NETWORK_PASSES_H
