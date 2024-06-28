#ifndef TENSOR_NETWORK_OPS
#define TENSOR_NETWORK_OPS

include "TensorNetwork/TensorNetworkTypes.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/CallInterfaces.td"
include "mlir/Bytecode/BytecodeOpInterface.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinTypes.td"
include "mlir/Interfaces/FunctionInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/BuiltinAttributes.td"
include "mlir/Interfaces/CastInterfaces.td"


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
        "Allows to specify a contraction manually."
    }]; 

    //Takes two tensors and returns a tensor
    let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs);
    let results = (outs AnyTensor:$result);
    let builders = [
        OpBuilder<(ins "Value":$lhs, "Value":$rhs), [{
            build($_builder, $_state, lhs, rhs);
        }]>,
    ];
}

// ================== TENSOR NETWORK OPERATIONS ==================
/* 
    The Dialect should be build like a graph, with nodes and edges.
 */
// Represents an index in the tensor network.
def IndexOp : TensorNetwork_Op<"index", [IsolatedFromAbove]>, Results<(outs IndexLabel)> {
    let summary = "Create an index for a tensor dimension";
    let description = [{
        //TODO
    }];

    let arguments = (ins I64Attr:$size, OptionalAttr<StrAttr>:$name);
    // let results = (outs IndexLabelType:$result);

    let builders = [
        OpBuilder<(ins "IntegerAttr":$size), [{
            build($_builder, $_state, size);
        }]>,
        OpBuilder<(ins "IntegerAttr":$size, "StringRef":$name), [{
            build($_builder, $_state, size, name);
        }]>,
    ];
}

// This represents a node in the tensor network. 
def TensorOp : TensorNetwork_Op<"tensor"> {
    let summary = "Create a tensor";
    let description = [{
        //TODO
    }]; 

    // let arguments = (ins F64ElementsAttr:$value, Variadic<IndexType>:$indices);
    let arguments = (ins F64ElementsAttr:$value, Variadic<IndexLabel>:$indices);
    let results = (outs F64Tensor:$result);

    let builders = [
    OpBuilder<(ins "DenseElementsAttr":$value, "ValueRange":$indices), [{
      build($_builder, $_state, value.getType(), value, indices);
    }]>,
    ];

}

// Should not be used, might remove later
// Represents an edge in the tensor network.
def ContractionEdgeOp : TensorNetwork_Op<"contraction_edge"> {
    let summary = "Specify a conection between two tensors in a tensor network";
    let description = [{
        //TODO
    }];

    let arguments = (ins AnyTensor:$lhs, AnyTensor:$rhs, ArrayAttr:$contracted_indices);
    let results = (outs AnyTensor:$result);
    let builders = [
        OpBuilder<(ins "Value":$lhs, "Value":$rhs, "ArrayAttr":$contracted_indices), [{
            build($_builder, $_state, lhs, rhs, contracted_indices);
        }]>,
    ];

    let assemblyFormat = "attr-dict  $lhs `:` type($lhs)  `with`  $rhs `:` type($rhs) `to` type($result)";
}




#endif // TENSOR_NETWORK_OPS
