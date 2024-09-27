#ifndef TENSOR_NETWORK_PASSES_H
#define TENSOR_NETWORK_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
    namespace tensor_network {

        #define GEN_PASS_DECL
        #include "TensorNetwork/TensorNetworkPasses.h.inc"


        enum class ContractionStrategy {
            Greedy,
            GreedyGrayKourtis
        };

        struct TensorNetworkNaiveLoweringOptions {
            ContractionStrategy contractionStrategy = ContractionStrategy::Greedy;
            bool enableRankSimplification = true;
            bool enableSlicing = false;
            double grayKourtisAlpha = 1.0;
            bool enableMultithreading = false;
        };


        std::unique_ptr<mlir::Pass> createTensorNetworkToLinalgLoweringPass();
        std::unique_ptr<mlir::Pass> createTensorNetworkNaiveLoweringPass();
        std::unique_ptr<mlir::Pass> createTensorNetworkNaiveLoweringPass(
            const TensorNetworkNaiveLoweringOptions &options);
        std::unique_ptr<mlir::Pass> createTensorNetworkToLLVMLoweringPass();

    } // namespace tensor_network
} // namespace mlir

#endif // TENSOR_NETWORK_PASSES_H
