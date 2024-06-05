#ifndef TENSOR_NETWORK_OPS_H
#define TENSOR_NETWORK_OPS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Attributes.h"
#include "TensorNetwork/TensorNetworkTypes.h"


#define GET_OP_CLASSES
#include "TensorNetwork/TensorNetworkOps.h.inc"

#endif // TENSOR_NETWORK_OPS_H