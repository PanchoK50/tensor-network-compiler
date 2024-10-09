# CompTN

## Instalation:
- First install and build llvm including MLIR
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

## System Overview:
![overview (1)](https://github.com/user-attachments/assets/8023d02d-8e87-43f3-9fef-3bcff2d3ba18)

## Passes:
- Rank Simplification
- Slicing
- Greedy path-search with:
    - Gray-Kourtis heuristic
    - FLOP heuristic
- Lowering passes

## Example:

```python

import tensor_network_ext as tn

mm = tn.ModuleManager()

index1 = mm.Index(2)
index2 = mm.Index(3)
index3 = mm.Index(4)

array1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
array2 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
array3 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])

tensor1 = mm.Tensor(array1, index1, index2)
tensor2 = mm.Tensor(array2, index1, index3)
tensor3 = mm.Tensor(array3, index2, index3)

mm.contract_multiple(tensor1, tensor2, tensor3)

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
