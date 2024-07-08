# tensor-network-compiler

## Currently working on

## TODO:

## Instalation:
- First install and build llvm. This will come with MLIR.
- Clone this repository and change these lines in the CMakeLists.txt file at the very top:
``` cmake
    #leave it like that if the llvm-project is in the same directory as this project
    set(MLIR_DIR "../llvm-project/build/lib/cmake/mlir")
    set(LLVM_DIR "../llvm-project/build/lib/cmake/llvm")
```
- Create a build directory and make the project:
```bash
    mkdir build \
    cd build \
    make
```

## Current Features:
- Generate C++ files from TableGen.
- Read MLIR (.mlir) files and output the Module. (e.g.`./build/bin/test_tensor_network test/mlir/mod_main_func.mlir`)
- Generate MLIR for the tensor network dialect via python keybindings (e.g. frontend/generate.py)
- Apply Lowerings (e.g `./build/bin/test_tensor_network --apply-lowerings test/mlir/mod_main_func.mlir`) 

## Tensor Network Dialect:
### How is the Tensor Network Dialect structured?
The Dialect is composed of Smart Indices that you declare, they represent a dimension and contain the size of the dimension. Then you can create Tensors using those Smart Indices. The Tensors then have the dimensions corresponding to these indices. If 2 Tensors share an index (or indices), they are contracted over the shared index/indices

## Usage:
- Create a build directory
- Inside the build directory you can run `cmake ..` and then `make`
- Create a python file including the shared libary tensor_network_ext (e.g frontend/generate.py)
- Save the output in an mlir file (working on connecting frontend with the backend directly rn)
- Run the executable with the path to the MLIR file as an argument (e.g. `./build/bin/test_tensor_network test/mlir/mod_main_func.mlir`)

## Example:
Command: `./build/bin/test_tensor_network --apply-lowerings test/mlir/mod_main_func.mlir`

## Helpful resources:
- [MLIR Troubleshooting] (https://makslevental.github.io/working-with-mlir/)
