# tensor-network-compiler

## Currently working on

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