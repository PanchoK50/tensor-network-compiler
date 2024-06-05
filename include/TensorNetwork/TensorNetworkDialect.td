#ifndef TENSOR_NETWORK_DIALECT
#define TENSOR_NETWORK_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/IR/BuiltinAttributes.td"

def TensorNetwork_Dialect : Dialect {
    let name = "tensor_network";

    let summary = "A dialect to for analyzing and optimizing the best ways to contract tensor networks.";

    let description = [{
        //TODO
    }];

    let cppNamespace = "::mlir::tensor_network";

    let useDefaultTypePrinterParser = 1;
    let extraClassDeclaration = [{
        void registerTypes();
    }];
}

class TensorNetwork_Op<string mnemonic, list<Trait> traits = []> :
        Op<TensorNetwork_Dialect, mnemonic, traits>;


#endif // TENSOR_NETWORK_DIALECT