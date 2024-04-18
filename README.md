# tensor-network-compiler

## Currently working on
- Data Type for Dummy Tensors (Tensors with given Rank and Shape, no values)
- First Hello World Program

## TODO:
- CMakeLists.txt
- Operations for: 
    - Tensor Contraction
    - Tensor Reshaping
    - Tensor Splitting

## Instalation:
- First install and build llvm. This will come with MLIR.
- Clone this repository and change these lines in the CMakeLists.txt file at the very top:
``` cmake
    #leave it like that if the llvm-project is in the same directory as this project
    set(MLIR_DIR "../llvm-project/build/lib/cmake/mlir")
    set(LLVM_DIR "../llvm-project/build/lib/cmake/llvm")
```

## Usage:
- Create a build directory
- Inside the build directory you can run `cmake ..` and then `make`