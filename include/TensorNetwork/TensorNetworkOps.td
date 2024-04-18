#ifndef TENSOR_NETWORK_OPS
#define TENSOR_NETWORK_OPS

include "TensorNetwork/TensorNetworkTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def ConstantOp : TensorNetwork_Op<"constant", []> {
    let summary = "Create a constant tensor";

    let arguments = (ins I32:$value);
    let results = (outs I32:$result);
}

#endif // TENSOR_NETWORK_OPS