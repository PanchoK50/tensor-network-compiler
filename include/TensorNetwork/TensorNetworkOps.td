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


// ============== TESTING OPERATIONS ===============

def ConstantTensorOp : TensorNetwork_Op<"constant_tensor"> {
    let summary = "Create a constant tensor";
    let description = [{
        //TODO
    }]; 

    let arguments = (ins F64ElementsAttr:$value);
    let results = (outs F64Tensor:$result);
}

//This is just a contraction edge, experimenting with nodes and edges for now
def ContractTensorsOp : TensorNetwork_Op<"contract"> {
    let summary = "Contract two tensors in a tensor network";
    let description = [{
        //TODO
    }]; 

    //Takes two tensors and returns a tensor
    let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
    let results = (outs AnyTensor:$result);
}

// ================== TENSOR NETWORK OPERATIONS ==================
/* 
    The Dialect should be build like a graph, with nodes and edges.
 */

// This represents a node in the tensor network. 
def TensorOp : TensorNetwork_Op<"tensor"> {
    let summary = "Create a tensor";
    let description = [{
        //TODO
    }]; 

    let arguments = (ins F64ElementsAttr:$value);
    let results = (outs F64Tensor:$result);

    let builders = [
    OpBuilder<(ins "DenseElementsAttr":$value), [{
      build($_builder, $_state, value.getType(), value);
    }]>,
    ];

}

// Represents an edge in the tensor network.
def ContractionEdgeOp : TensorNetwork_Op<"contraction_edge"> {
    let summary = "Specify a conection between two tensors in a tensor network";
    let description = [{
        //TODO
    }];

    // let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs, ArrayAttr:$contracted_indices);
    // let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
    // let results = (outs AnyTensor:$result);
    // let builders = [
    //     OpBuilder<(ins "Value":$lhs, "Value":$rhs), [{
    //         build($_builder, $_state, lhs, rhs);
    //     }]>,
    // ];

    let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs, ArrayAttr:$contracted_indices);
    let results = (outs AnyTensor:$result);
    let builders = [
        OpBuilder<(ins "Value":$lhs, "Value":$rhs, "ArrayAttr":$contracted_indices), [{
            build($_builder, $_state, lhs, rhs, contracted_indices);
        }]>,
    ];

    // let assemblyFormat = "$lhs $rhs attr-dict type($lhs) `,` type($rhs) `to` type($result)";
    let assemblyFormat = "attr-dict  $lhs `:` type($lhs)  `with`  $rhs `:` type($rhs) `to` type($result)";
}


#endif // TENSOR_NETWORK_OPS