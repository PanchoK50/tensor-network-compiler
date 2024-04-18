#ifndef TENSOR_NETWORK_PASS
#define TENSOR_NETWORK_PASS

include "mlir/Pass/PassBase.td"

def DummyPass: Pass<"", ""> {
    let summary = "This is just a dummy pass, for testing";
}

#endif // TENSOR_NETWORK_PASS