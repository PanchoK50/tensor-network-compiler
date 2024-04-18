#ifndef TENSOR_NETWORK_PASSES_H
#define TENSOR_NETWORK_PASSES_H

#include "TensorNetwork/TensorNetworkDialect.h"
#include "TensorNetwork/TensorNetworkOps.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace tensor_network {

#define GEN_PASS_DECL
#include "TensorNetwork/TensorNetworkPasses.h.inc"

#define GEN_PASS_REGISTRATION
#include "TensorNetwork/TensorNetworkPasses.h.inc"

} // namespace tensor_network
} // namespace mlir

#endif // TENSOR_NETWORK_PASSES_H
