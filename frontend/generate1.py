import sys, os
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
lib_dir = os.path.join(script_dir, '../build/lib/TensorNetwork')
sys.path.append(lib_dir)

import tensor_network_ext as tn

mm = tn.ModuleManager()

index1 = mm.Index(2)

array1 = np.array([1.0, 2.0])

tensor1 = mm.Tensor(array1, index1)
tensor2 = mm.Tensor(array1, index1)

contracted = mm.contract(tensor1, tensor2)

run_result = mm.run()

print(run_result)

