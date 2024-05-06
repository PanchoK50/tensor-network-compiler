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

// def TensorShapeType : Type<"TensorShapeType", "tensor_shape"> {
//   let summary = "A shape of a tensor";
//   let description = [{
//     //TODO
//   }];

//   let parameters = (ins "dims":$dims);
//   let assemblyFormat = "`tensor_shape` `<` $dims `>`";
// }

// def TensorIndicesType : Type<"TensorIndicesType", "tensor_indices"> {
//   let summary = "A tensor indices type";
//   let description = [{
//     A tensor indices type represents the indices of a tensor.
//   }];

//   let parameters = (ins "indices":$indices);

//   let assemblyFormat = "`tensor_indices` `<` $indices `>`";
// }

// def TensorNode : TensorNetwork_Type<"TensorNode", "tensor_node"> {
//     let summary = "A tensor node in a tensor network";
//     let description = "A tensor node in a tensor network";

//     let parameters = (shape : TensorShapeType, indices : TensorIndicesType);
//     let assemblyFormat = "`tensor_node` `<` $shape `>` `(` $indices `)`";
// }


// class TensorShapeAttr : AttrDef<TensorNetwork_Dialect, "TensorShape"> {
//   let mnemonic = "tensor_shape";
// }

// def TensorShape : TensorShapeAttr {
//   let summary = "A shape of a tensor";
//   let description = [{
//     //TODO
//   }];

//   let parameters = (ins "dims":$dims);
//   let assemblyFormat = "`tensor_shape` `<` $dims `>`";
// }

// class TensorIndicesAttr : AttrDef<TensorNetwork_Dialect, "TensorIndices"> {
//   let mnemonic = "tensor_indices";
// }

// def TensorIndices : TensorIndicesAttr {
//   let summary = "A tensor indices attribute";
//   let description = [{
//     A tensor indices attribute represents the indices of a tensor.
//   }];

//   let parameters = (ins "indices":$indices);

//   let assemblyFormat = "`tensor_indices` `<` $indices `>`";
// }
#endif // TENSOR_NETWORK_TYPES

// #ifndef TENSOR_NETWORK_TYPES
// #define TENSOR_NETWORK_TYPES

// include "mlir/IR/AttrTypeBase.td"
// include "TensorNetwork/TensorNetworkDialect.td"

// class TensorNetwork_Type<string name, string typeMnemonic, list<Trait> traits = []>
//     : TypeDef<TensorNetwork_Dialect, name, traits> {
//   let mnemonic = typeMnemonic;
// }

// def DummyTensorType: TensorNetwork_Type<"DummyTensor", "dummy_tensor"> {
//     let summary = "A dummy tensor used to optimize tensor network contraction operations";
//     let description = "The Dummy Tensor is for testing purposes right now. The purpose of it is to store shape and rank";
// }

// def TensorShapeType : TensorNetwork_Type<"TensorShapeType", "tensor_shape"> {
//   let summary = "A shape of a tensor";
//   let description = [{
//     //TODO
//   }];

//   let parameters = (ins "dims":$dims);
//   let assemblyFormat = "`tensor_shape` `<` $dims `>`";
// }

// def TensorIndicesType : TensorNetwork_Type<"TensorIndicesType", "tensor_indices"> {
//   let summary = "A tensor indices type";
//   let description = [{
//     A tensor indices type represents the indices of a tensor.
//   }];

//   let parameters = (ins "indices":$indices);

//   let assemblyFormat = "`tensor_indices` `<` $indices `>`";
// }

// def TensorNode : TensorNetwork_Type<"TensorNode", "tensor_node"> {
//     let summary = "A tensor node in a tensor network";
//     let description = "A tensor node in a tensor network";

//     let parameters = (ins TensorShapeType:$shape, TensorIndicesType:$indices);
//     let assemblyFormat = "`tensor_node` `<` $shape `>` `(` $indices `)`";
// }


// class TensorShapeAttr : AttrDef<TensorNetwork_Dialect, "TensorShape"> {
//   let mnemonic = "tensor_shape";
// }

// def TensorShape : TensorShapeAttr {
//   let summary = "A shape of a tensor";
//   let description = [{
//     //TODO
//   }];

//   let parameters = (ins "dims":$dims);
//   let assemblyFormat = "`tensor_shape` `<` $dims `>`";
// }

// class TensorIndicesAttr : AttrDef<TensorNetwork_Dialect, "TensorIndices"> {
//   let mnemonic = "tensor_indices";
// }

// def TensorIndices : TensorIndicesAttr {
//   let summary = "A tensor indices attribute";
//   let description = [{
//     A tensor indices attribute represents the indices of a tensor.
//   }];

//   let parameters = (ins "indices":$indices);

//   let assemblyFormat = "`tensor_indices` `<` $indices `>`";
// }
// #endif // TENSOR_NETWORK_TYPES