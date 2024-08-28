#include "TensorNetwork/TensorNetworkTypes.h"

#include "TensorNetwork/TensorNetworkDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir::tensor_network;

#define GET_TYPEDEF_CLASSES
#include "TensorNetwork/TensorNetworkOpsTypes.cpp.inc"

void TensorNetworkDialect::registerTypes() {
    addTypes<
#define GET_TYPEDEF_LIST
#include "TensorNetwork/TensorNetworkOpsTypes.cpp.inc"
    >();
}




