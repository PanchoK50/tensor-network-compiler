import sys, os

script_dir = os.path.dirname(os.path.abspath(__file__))
lib_dir = os.path.join(script_dir, '../build/lib/TensorNetwork')
sys.path.append(lib_dir)

import tensor_network_ext as tn
import numpy as np

mm = tn.ModuleManager()
indexI = mm.Index(6)
indexJ = mm.Index(6)
indexK = mm.Index(6)
indexL = mm.Index(6)

array1 = np.random.rand(6, 6, 6)
array2 = np.random.rand(6, 6)
array3 = np.random.rand(6, 6)

tensorA = mm.Tensor(array1, indexI, indexJ, indexK)
tensorB = mm.Tensor(array2, indexJ, indexL)
tensorC = mm.Tensor(array3, indexK, indexL)

mm.contract_multiple(tensorA, tensorB, tensorC)
result = mm.run()
print("CompTN Result: " + str(result))
