#ifndef TENSOR_NETWORK_TYPES
#define TENSOR_NETWORK_TYPES

include "mlir/IR/AttrTypeBase.td"
include "TensorNetwork/TensorNetworkDialect.td"

class TensorNetwork_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<TensorNetwork_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def DummyTensorType: TensorNetwork_Type<"DummyTensor", "dummy_tensor"> {
    let summary = "A dummy tensor used to optimize tensor network contraction operations";
    let description = "The Dummy Tensor is for testing purposes right now. The purpose of it is to store shape and rank";
}

def IndexLabelType : TensorNetwork_Type<"IndexLabel", "indexlabel"> {
    let summary = "An index label used in tensor network operations";
    let description = "An index label used in tensor network operations";

    let parameters = (ins
        "mlir::IntegerAttr":$size,
        "mlir::StringAttr":$name
    );

    let assemblyFormat = "`<` $size `,` $name `>`";
}

def TensorWithIndicesType : TensorNetwork_Type<"TensorWithIndices", "tensor_with_indices"> {
       let summary = "Type representing a tensor with associated indices";
       let description = [{
           A tensor with indices type represents a tensor along with its associated indices.
       }];

       let parameters = (ins
           "mlir::Type":$tensorType,
           "mlir::ArrayAttr":$indices
       );

       let assemblyFormat = "`<` $tensorType `,` $indices `>`";
}

def TNIsTensorWithIndicesTypePred : CPred<"$_self.isa<TensorWithIndicesType>()">;
def TensorWithIndices : Type<TNIsTensorWithIndicesTypePred, "tensorwithindices">;

//Similar to COMET
def TNIsIndexLabelTypePred : CPred<"$_self.isa<IndexLabelType>()">;
def IndexLabel : Type<TNIsIndexLabelTypePred, "indexlabel">;

#endif // TENSOR_NETWORK_TYPES
