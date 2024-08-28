import sys, os
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
lib_dir = os.path.join(script_dir, '../build/lib/TensorNetwork')
sys.path.append(lib_dir)

import tensor_network_ext as tn

mm = tn.ModuleManager()

index1 = mm.Index(2)
index2 = mm.Index(3)
index3 = mm.Index(4)

# array1 = np.random.randn(2, 3)
# array2 = np.random.randn(2, 4)

array1 = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
array2 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])

tensor1 = mm.Tensor(array1, index1, index2)
tensor2 = mm.Tensor(array2, index1, index3)
tensor3 = mm.Tensor(array1, index1, index2)

contracted = mm.contract(tensor1, tensor2)

# You can now use the result of the contraction in further operations
# array3 = np.random.randn(3, 4)

# Change this to floating point numbers
array3 = np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]])
tensor3 = mm.Tensor(array3, index2, index3)

final_result = mm.contract(contracted, tensor3)

print("Module:")
mm.dump()


print("Result:")
run_result = mm.run()

print(run_result)

# mm.dump()
