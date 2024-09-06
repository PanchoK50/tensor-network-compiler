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

// ================== TENSOR NETWORK OPERATIONS ==================

def TransposeOp : TensorNetwork_Op<"transpose"> {
    let summary = "Transpose a tensor along specified indices";
    let description = [{
        Transposes a tensor along specified indices and returns a new tensor.
        The idea is that the indices are permuted such that the specified indices
        allow for less cache misses when iterating over the tensor..
    }];

    let arguments = (ins
        TensorWithIndicesType:.$tensor,
        Variadic<IndexLabel>:$indices
    );

    let results = (outs
        TensorWithIndicesType:$result
    );
}

def AddOp : TensorNetwork_Op<"add"> {
    let summary = "Add two tensors";
    let description = [{
        Adds two tensors and returns the result.
    }];

    let arguments = (ins
        TensorWithIndicesType:$lhs,
        TensorWithIndicesType:$rhs
    );

    let results = (outs
        TensorWithIndicesType:$result
    );
}

def ConcatOp : TensorNetwork_Op<"concat"> {
    let summary = "Concatenate two tensors along specified indices";
    let description = [{
        Concatenates two tensors along specified indices and returns a new tensor with the remaining indices.
    }];

    let arguments = (ins
        TensorWithIndicesType:$lhs,
        TensorWithIndicesType:$rhs
    );

    let results = (outs
        TensorWithIndicesType:$result
    );
}

def TensorDeclOp : TensorNetwork_Op<"tensor_decl"> {
    let summary = "Declare a tensor with indices";
    let description = [{
        //TODO
    }];

    let arguments = (ins Variadic<IndexLabel>:$indices);
    let results = (outs TensorWithIndicesType:$result);
}

def ContractTensorsOp : TensorNetwork_Op<"contract"> {
    let summary = "Contract two tensors along specified indices";
    let description = [{
        Contracts two tensors along specified indices and returns a new tensor with the remaining indices.
    }];

    let arguments = (ins
        TensorWithIndicesType:$lhs,
        TensorWithIndicesType:$rhs
    );

    let results = (outs
        TensorWithIndicesType:$result
    );

}

def ContractMultipleTensorsOp : TensorNetwork_Op<"contract_multiple"> {
    let summary = "Contract multiple tensors along specified indices";
    let description = [{
        Contracts multiple tensors along specified indices and returns a new tensor with the remaining indices.
    }];

    let arguments = (ins
        Variadic<TensorWithIndicesType>:$tensors
    );

    let results = (outs
        TensorWithIndicesType:$result
    );

}

// Represents an index in the tensor network.
def IndexOp : TensorNetwork_Op<"index", [IsolatedFromAbove, Pure]>, Results<(outs IndexLabel)> {
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

def TensorOp : TensorNetwork_Op<"tensor"> {
    let summary = "Create a tensor with indices with constant value as attribute";
    let description = [{
        //TODO
    }]; 

    let arguments = (ins F64ElementsAttr:$value, Variadic<IndexLabel>:$indices);
    let results = (outs TensorWithIndicesType:$result);
}

def TensorFromValueOp : TensorNetwork_Op<"tensor_from_value"> {
    let summary = "Create a tensor with indices from an SSA value";
    let description = [{
        This operation creates a tensor with indices from an SSA value.
    }];

    let arguments = (ins
        AnyTensor:$value
    );
    let results = (outs TensorWithIndicesType:$result);
}

#endif // TENSOR_NETWORK_OPS
