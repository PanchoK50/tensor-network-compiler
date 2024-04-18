#ifndef TENSOR_NETWORK_TYPES
#define TENSOR_NETWORK_TYPES

include "mlir/IR/AttrTypeBase.td"
include "TensorNetwork/TensorNetworkDialect.td"

class TensorNetwork_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<TensorNetwork_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

def TensorNetwork_DummyTensor_Type : TensorNetwork_Type<"DummyTensor", "dummy_tensor"> {
    let summary = "A dummy tensor used to optimize tensor network contraction operations";
    let description = "//TODO";
}

#endif // TENSOR_NETWORK_TYPES