#ifndef TENSOR_NETWORK_OPS
#define TENSOR_NETWORK_OPS

include "TensorNetwork/TensorNetworkTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Bytecode/BytecodeOpInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"


def ConstantTensorOp : TensorNetwork_Op<"constant_tensor"> {
    let summary = "Create a constant tensor";
    let description = [{
        //TODO
    }]; 

    let arguments = (ins F64ElementsAttr:$value);
    let results = (outs F64Tensor:$result);
}

def ContractTensorsOp : TensorNetwork_Op<"contract"> {
    let summary = "Contract two tensors in a tensor network";
    let description = [{
        //TODO
    }]; 

    //Takes two tensors and returns a tensor
    let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
    let results = (outs AnyTensor:$result);
}

#endif // TENSOR_NETWORK_OPS