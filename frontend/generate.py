# from mlir.ir import Context, InsertionPoint, Location, Module, Operation
import sys, os

script_dir = os.path.dirname(os.path.abspath(__file__)) # Get the directory where this script is located
lib_dir = os.path.join(script_dir, '../build/lib/TensorNetwork') # Construct the path to the directory containing the shared library
sys.path.append(lib_dir) # Add this directory to sys.path

import tensor_network_ext as tn
import numpy as np

mm = tn.ModuleManager()

index1 = mm.Index(2, "i")
index2 = mm.Index(3, "j")
index3 = mm.Index(4, "k")
index4 = mm.Index(5, "l")

# Create a numpy array to use for creating a tensor operation
# array1 = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
# array2 = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float64)
# array3 = np.random.randn(2, 2, 2)

array1 = np.random.randn(2, 3)
array2 = np.random.randn(2, 4)

# array1 = np.random.randn(2, 3, 5)
# array2 = np.random.randn(3, 5, 4)


# Create a tensor operation with the numpy array and the index operations
tensor1 = mm.Tensor(array1, index1, index2)
tensor2 = mm.Tensor(array2, index1, index3)
# tensor3 = mm.Tensor(array2, index2, index3)
# tensor1 = mm.Tensor(array1, index1, index2, index4)
# tensor2 = mm.Tensor(array2, index2, index3, index4)

mm.contract(tensor1, tensor2)

# Interface to determine contraction order

# Dump the current state of the module again to verify the tensor operation was added
mm.dump()
    
