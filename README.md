# tensor-network-compiler

## Currently working on
- Data Type for Dummy Tensors (Tensors with given Rank and Shape, no values).

## TODO:
- Data Type for Tensor Network.
- Operations for Tensors (Contraction of tensors and subsequent addition of result to tensor network).

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

## Usage:
- Create a build directory
- Inside the build directory you can run `cmake ..` and then `make`