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
- Import the tensor_network_ext library into python and execute the contractions via JIT

## Tensor Network Dialect:
### How is the Tensor Network Dialect structured?
The Dialect is composed of Smart Indices that you declare, they represent a dimension and contain the size of the dimension. Then you can create Tensors using those Smart Indices. The Tensors then have the dimensions corresponding to these indices. If 2 Tensors share an index (or indices), they are contracted over the shared index/indices

## Usage:
- Create a build directory
- Inside the build directory you can run `cmake ..` and then `make`
- Create a python file including the shared libary tensor_network_ext (e.g frontend/generate.py)
- Run the computation with the run() function

## Example:

```python

import tensor_network_ext as tn

mm = tn.ModuleManager()

index1 = mm.Index(2)
index2 = mm.Index(3)
index3 = mm.Index(4)

array1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
array2 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])

tensor1 = mm.Tensor(array1, index1, index2)
tensor2 = mm.Tensor(array2, index1, index3)
tensor3 = mm.Tensor(array1, index1, index2)

contracted = mm.contract(tensor1, tensor2)

array3 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])
tensor3 = mm.Tensor(array3, index2, index3)

final_result = mm.contract(contracted, tensor3)

print("Module:")
mm.dump()

print("Result:")
run_result = mm.run()
print(run_result)


```

## Helpful resources:
- [MLIR Troubleshooting] (https://makslevental.github.io/working-with-mlir/)
- [Writing a Kernel] (https://medium.com/@hxu296/a-trip-to-kernels-understanding-pytorchs-internal-architecture-fc955aafd54c)
- [Writing a Kernel 2] (https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/README.md)
