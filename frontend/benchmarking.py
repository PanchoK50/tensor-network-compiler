import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
lib_dir = os.path.join(script_dir, '../build/lib/TensorNetwork')
sys.path.append(lib_dir)

import time
import numpy as np
import tensor_network_ext as tn

def parse_einsum(einsum_notation):
    input_output = einsum_notation.split('->')
    if len(input_output) != 2:
        raise ValueError("Invalid einsum notation. Must contain '->'")

    input_tensors = input_output[0].split(',')
    output_indices = input_output[1].strip()

    tensor_indices = [tensor.strip() for tensor in input_tensors]

    return tensor_indices, output_indices

def create_tensor_network_from_sizes(einsum_notation, *index_sizes):
    mm = tn.ModuleManager()
    tensor_indices, output_indices = parse_einsum(einsum_notation)

    unique_indices = list(dict.fromkeys(''.join(tensor_indices)))

    if len(index_sizes) != len(unique_indices):
        raise ValueError("Number of index sizes doesn't match the number of unique indices in the einsum notation")

    size_dict = dict(zip(unique_indices, index_sizes))
    index_dict = {}
    tensors = []

    for tensor_index in tensor_indices:
        tensor_indices_list = []
        shape = []
        for idx in tensor_index:
            if idx not in index_dict:
                index_dict[idx] = mm.Index(size_dict[idx])
            tensor_indices_list.append(index_dict[idx])
            shape.append(size_dict[idx])
        
        array = np.random.rand(*shape)
        tensor = mm.Tensor(array, *tensor_indices_list)
        tensors.append(tensor)

    mm.contract_multiple(*tensors)
    return mm

def create_tensor_network_from_arrays(einsum_notation, *arrays):
    mm = tn.ModuleManager()
    tensor_indices, output_indices = parse_einsum(einsum_notation)

    if len(arrays) != len(tensor_indices):
        raise ValueError("Number of arrays doesn't match the einsum notation")

    index_dict = {}
    tensors = []

    for tensor_index, array in zip(tensor_indices, arrays):
        tensor_indices_list = []
        for idx, size in zip(tensor_index, array.shape):
            if idx not in index_dict:
                index_dict[idx] = mm.Index(size)
            tensor_indices_list.append(index_dict[idx])
        
        tensor = mm.Tensor(array, *tensor_indices_list)
        tensors.append(tensor)

    mm.contract_multiple(*tensors)
    return mm

def numpy_einsum(einsum_notation, *arrays):
    return np.einsum(einsum_notation, *arrays)

def create_numpy_arrays(einsum_notation, *index_sizes):
    tensor_indices, _ = parse_einsum(einsum_notation)
    
    # Get unique indices from input tensors only
    unique_indices = list(dict.fromkeys(''.join(tensor_indices)))

    if len(index_sizes) != len(unique_indices):
        raise ValueError("Number of index sizes doesn't match the number of unique indices in the einsum notation")

    size_dict = dict(zip(unique_indices, index_sizes))
    
    arrays = []
    for tensor_index in tensor_indices:
        shape = [size_dict[idx] for idx in tensor_index]
        arrays.append(np.random.rand(*shape))
    
    return arrays

def benchmark_tensor_network(einsum_notation, *index_sizes):
    print(f"\nBenchmarking: {einsum_notation}")
    print(f"Index sizes: {index_sizes}")

    # Create NumPy arrays
    numpy_arrays = create_numpy_arrays(einsum_notation, *index_sizes)

    # Tensor Network benchmark
    creation_start = time.time()
    mm = create_tensor_network_from_arrays(einsum_notation, *numpy_arrays)
    creation_end = time.time()
    creation_time = creation_end - creation_start

    compile_start = time.time()
    mm.compile()
    compile_end = time.time()
    compile_time = compile_end - compile_start

    run_start = time.time()
    result = mm.run_compiled()
    run_end = time.time()
    run_time = run_end - run_start

    print(f"Tensor Network:")
    print(f"  Creation time:    {creation_time:.6f} seconds")
    print(f"  Compilation time: {compile_time:.6f} seconds")
    print(f"  Run time:         {run_time:.6f} seconds")
    print(f"  Total time:       {creation_time + compile_time + run_time:.6f} seconds")
    print(f"  Result shape:     {result.shape}")

    # NumPy benchmark
    numpy_start = time.time()
    numpy_result = numpy_einsum(einsum_notation, *numpy_arrays)
    numpy_end = time.time()
    numpy_time = numpy_end - numpy_start

    print(f"NumPy:")
    print(f"  Total time: {numpy_time:.6f} seconds")
    print(f"  Result shape: {numpy_result.shape}")

    # Verify results
    if np.allclose(result, numpy_result):
        print("Results match!")
    else:
        print("WARNING: Results do not match!")

    print("---")

def main():
    # Run benchmarks

    # Important: No hyperindices

    benchmark_tensor_network("ij,jk->ik", 10, 20, 30)
    benchmark_tensor_network("abc,cd,db->a", 5, 6, 7, 8)
    benchmark_tensor_network("abcd,dcba->", 2, 3, 4, 5)
    benchmark_tensor_network("ijk,jkl,lm->im", 10, 15, 20, 25, 10)

if __name__ == "__main__":
    main()
