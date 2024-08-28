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

tensor1 = mm.Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), index1, index2)
tensor2 = mm.Tensor(np.array([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]), index1, index3)

contracted = mm.contract(tensor1, tensor2)

print("Result:")
run_result = mm.run()

print(run_result)
