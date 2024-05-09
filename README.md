# tensor-network-compiler

## Currently working on

  

## TODO:
- [ ] Lowering of TensorNetwork Dialect to Standard Dialect
- [ ] Very simple optimization passes for testing

## Instalation:
- First install and build llvm. This will come with MLIR.
- Clone this repository and change these lines in the CMakeLists.txt file at the very top:
``` cmake
    #leave it like that if the llvm-project is in the same directory as this project
    set(MLIR_DIR "../llvm-project/build/lib/cmake/mlir")
    set(LLVM_DIR "../llvm-project/build/lib/cmake/llvm")
```

## Current Features:
- Generate C++ files from TableGen.
- Read MLIR (.mlir) files and output the Module. (e.g.`./build/bin/test_tensor_network test/mlir/first_test.mlir`)
- Generate MLIR for the tensor network dialect.
- Apply Lowerings (e.g `./build/bin/test_tensor_network --apply-lowerings test/mlir/ten_tensors.mlir`) (No Lowerings are implemented yet but the infrastructure should work now)

## Tensor Network Dialect:
### How is the Tensor Network Dialect structured?
The Dialect is designed to be very close to the tensor network graphical notation. It allows to (first) declare all the tensors involved and (second) declare all the edges between the tensors.

## Usage:
- Create a build directory
- Inside the build directory you can run `cmake ..` and then `make`
- Create an MLIR file with the desired operations
- Run the executable with the path to the MLIR file as an argument (e.g. `./build/bin/test_tensor_network test/mlir/first_test.mlir`)

## Example:
Command: `./build/bin/test_tensor_network test/mlir/tensor_network_test.mlir`

Output: 
```mlir
module {
  func.func @main() -> tensor<f64> {
    %0 = "tensor_network.constant_tensor"() <{value = dense<1.000000e+00> : tensor<1xf64>}> : () -> tensor<f64>
    %1 = "tensor_network.constant_tensor"() <{value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00]> : tensor<3xf64>}> : () -> tensor<f64>
    %2 = "tensor_network.constant_tensor"() <{value = dense<[1.000000e+00, 2.000000e+00, 4.000000e+00]> : tensor<3xf64>}> : () -> tensor<f64>
    return %0 : tensor<f64>
  }
}
 ```

## Generate MLIR:

- In order to generate MLIR code, you can use OpBuilders like in `generate.cpp`
- (Of course you can roundtrip this generated mlir into a module, like described in Usage)

`./build/bin/generate_mlir_ir`: 
```mlir
module {
  %0 = "tensor_network.tensor"() <{value = dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf64>}> : () -> tensor<2x2xf64>
  %1 = "tensor_network.tensor"() <{value = dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf64>}> : () -> tensor<2x2xf64>
  %2 = "tensor_network.tensor"() <{value = dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf64>}> : () -> tensor<2x2xf64>
  %3 = "tensor_network.tensor"() <{value = dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf64>}> : () -> tensor<2x2xf64>
  %4 = "tensor_network.tensor"() <{value = dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf64>}> : () -> tensor<2x2xf64>
  %5 = "tensor_network.tensor"() <{value = dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf64>}> : () -> tensor<2x2xf64>
  %6 = "tensor_network.tensor"() <{value = dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf64>}> : () -> tensor<2x2xf64>
  %7 = "tensor_network.tensor"() <{value = dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf64>}> : () -> tensor<2x2xf64>
  %8 = "tensor_network.tensor"() <{value = dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf64>}> : () -> tensor<2x2xf64>
  %9 = "tensor_network.tensor"() <{value = dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf64>}> : () -> tensor<2x2xf64>
  %10 = "tensor_network.contraction_edge"(%0, %1) : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
  %11 = "tensor_network.contraction_edge"(%1, %2) : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
  %12 = "tensor_network.contraction_edge"(%2, %3) : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
  %13 = "tensor_network.contraction_edge"(%3, %4) : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
  %14 = "tensor_network.contraction_edge"(%4, %5) : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
  %15 = "tensor_network.contraction_edge"(%5, %6) : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
  %16 = "tensor_network.contraction_edge"(%6, %7) : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
  %17 = "tensor_network.contraction_edge"(%7, %8) : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
  %18 = "tensor_network.contraction_edge"(%8, %9) : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
}

 ```