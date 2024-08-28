#include "TensorNetwork/TensorNetworkPasses.h"
#include "TensorNetwork/TensorNetworkPasses.h"
#include "TensorNetwork/TensorNetworkOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"

namespace mlir::tensor_network {
#include "TensorNetwork/TensorNetworkPasses.h.inc"
} //namespace mlir::tensor_network
