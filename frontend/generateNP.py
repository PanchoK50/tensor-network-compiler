import numpy as np

# Set NumPy to use a higher precision
np.set_printoptions(precision=20)

def numpy_contract(tensor1, tensor2):
    # This performs a contraction over all shared indices
    # In this case, it's equivalent to a dot product
    return np.dot(tensor1, tensor2)

# Mirror the operations in your MLIR-based code
array1 = np.array([1.0, 2.0], dtype=np.float64)

# Perform the contraction
numpy_result = numpy_contract(array1, array1)

print("NumPy Result:")
print(numpy_result)
print("NumPy Result in hex:")
print(numpy_result.hex())


def numpy_contract_tensor(tensor1, tensor2, axes):
    return np.tensordot(tensor1, tensor2, axes=axes)

# Mirror the operations in your MLIR-based code
array1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float64)
array2 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], dtype=np.float64)
array3 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]], dtype=np.float64)

# Perform the first contraction (tensor1 and tensor2)
# Contract over the first axis (index1)
contracted = numpy_contract_tensor(array1, array2, axes=([0], [0]))

# Perform the second contraction (result of first contraction with tensor3)
# Contract over the remaining axes
final_result = numpy_contract_tensor(contracted, array3, axes=([0, 1], [0, 1]))

print("NumPy Result:")
print(final_result)

# Create tensors
tensor1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
tensor2 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])

# Perform contraction
# This is equivalent to contracting along the first axis (index1 in the original code)
contracted = np.einsum('ij,ik->jk', tensor1, tensor2)

print("Result:")
print(contracted)

# If you want to see the shape of the result
print("\nShape of the result:")
print(contracted.shape)
