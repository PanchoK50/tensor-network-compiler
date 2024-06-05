# from mlir.ir import Context, InsertionPoint, Location, Module, Operation
import sys, os

script_dir = os.path.dirname(os.path.abspath(__file__)) # Get the directory where this script is located
lib_dir = os.path.join(script_dir, '../build/lib/TensorNetwork') # Construct the path to the directory containing the shared library
sys.path.append(lib_dir) # Add this directory to sys.path

import tensor_network_ext as tn

mdl = tn.ModuleManager()

index1 = mdl.Index(4, "i")
index2 = mdl.Index(4, "j")

mdl.dump()
    
