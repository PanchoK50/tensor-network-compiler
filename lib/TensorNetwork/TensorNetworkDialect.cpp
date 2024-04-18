#include "TensorNetwork/TensorNetworkTypes.h"
#include "TensorNetwork/TensorNetworkOps.h"
#include "TensorNetwork/TensorNetworkTypes.h"
#include "TensorNetwork/TensorNetworkDialect.h"

using namespace mlir;
using namespace mlir::tensor_network;

#include "TensorNetwork/TensorNetworkOpsDialect.cpp.inc"

void mlir::tensor_network::TensorNetworkDialect::initialize() {
    addOperations<
    #define GET_OP_LIST
    #include "TensorNetwork/TensorNetworkOps.cpp.inc"
    >();
    registerTypes();
}