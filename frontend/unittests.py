import unittest
import numpy as np
import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
lib_dir = os.path.join(script_dir, '../build/lib/TensorNetwork')
sys.path.append(lib_dir)

import tensor_network_ext as tn

class TestTensorNetwork(unittest.TestCase):
    
    def setUp(self):
        self.mm = tn.ModuleManager()

    def test_simple_contraction(self):
        print("\n--- Test Simple Contraction ---")
        index1 = self.mm.Index(2)
        index2 = self.mm.Index(2)
        index3 = self.mm.Index(2)

        matrix1 = np.array([[1, 2], [3, 4]], dtype=np.float64)
        matrix2 = np.array([[5, 6], [7, 8]], dtype=np.float64)

        tensor1 = self.mm.Tensor(matrix1, index1, index2)
        tensor2 = self.mm.Tensor(matrix2, index2, index3)

        self.mm.contract_multiple(tensor1, tensor2)
        result = self.mm.run()

        expected = np.einsum('ij,jk->ik', matrix1, matrix2)
        
        print("CompTN result:")
        print(result)
        print("NumPy result:")
        print(expected)
        
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_large_tensor_network(self):
        print("\n--- Test Large Tensor Network ---")
        n = 5  # number of tensors
        d = 2  # number of dimensions

        indices = [self.mm.Index(d) for _ in range(n+1)]
        tensors = []
        numpy_tensors = []

        for i in range(n):
            tensor_data = np.random.rand(d, d)
            tensor = self.mm.Tensor(tensor_data, indices[i], indices[i+1])
            tensors.append(tensor)
            numpy_tensors.append(tensor_data)

        # Contract all tensors
        self.mm.contract_multiple(*tensors)
        final_result = self.mm.run()

        # Calculate expected result using numpy
        expected = np.eye(d)
        for tensor_data in numpy_tensors:
            expected = np.einsum('ij,jk->ik', expected, tensor_data)

        print("CompTN result:")
        print(final_result)
        print("NumPy result:")
        print(expected)
        
        np.testing.assert_allclose(final_result, expected, rtol=1e-5)

    def test_quantum_circuit_simulation(self):
        print("\n--- Test Quantum Circuit Simulation ---")
        n_qubits = 3
        indices = [self.mm.Index(2) for _ in range(n_qubits)]

        # Initialize state |000>
        state = np.zeros(2**n_qubits)
        state[0] = 1
        initial_state = self.mm.Tensor(state.reshape([2]*n_qubits), *indices)

        # Apply Hadamard gate to first qubit
        h_gate = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
        h_tensor = self.mm.Tensor(h_gate, indices[0], self.mm.Index(2))

        self.mm.contract_multiple(h_tensor, initial_state)
        result = self.mm.run()

        expected = np.kron(h_gate, np.eye(4)).dot(state)
        
        print("CompTN result:")
        print(result.flatten())
        print("NumPy result:")
        print(expected)
        
        np.testing.assert_allclose(result.flatten(), expected, rtol=1e-5)

    def test_multiple_contractions(self):
        print("\n--- Test Multiple Contractions ---")
        index1 = self.mm.Index(2)
        index2 = self.mm.Index(2)
        index3 = self.mm.Index(2)
        index4 = self.mm.Index(2)

        matrix1 = np.array([[1, 2], [3, 4]], dtype=np.float64)
        matrix2 = np.array([[5, 6], [7, 8]], dtype=np.float64)
        matrix3 = np.array([[9, 10], [11, 12]], dtype=np.float64)

        tensor1 = self.mm.Tensor(matrix1, index1, index2)
        tensor2 = self.mm.Tensor(matrix2, index2, index3)
        tensor3 = self.mm.Tensor(matrix3, index3, index4)

        self.mm.contract_multiple(tensor1, tensor2, tensor3)
        result = self.mm.run()

        expected = np.einsum('ij,jk,kl->il', matrix1, matrix2, matrix3)
        
        print("CompTN result:")
        print(result)
        print("NumPy result:")
        print(expected)
        
        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_intertwined_tensors(self):
        print("\n--- Test Intertwined Tensors ---")
        index1 = self.mm.Index(2)
        index2 = self.mm.Index(3)
        index3 = self.mm.Index(4)
        index4 = self.mm.Index(2)
        index5 = self.mm.Index(2)

        tensor1_data = np.random.rand(2, 3, 4)
        tensor2_data = np.random.rand(3, 4, 2)
        tensor3_data = np.random.rand(2, 2)
        
        tensor1 = self.mm.Tensor(tensor1_data, index1, index2, index3)
        tensor2 = self.mm.Tensor(tensor2_data, index2, index3, index4)
        tensor3 = self.mm.Tensor(tensor3_data, index4, index5)

        self.mm.contract_multiple(tensor1, tensor2, tensor3)
        result = self.mm.run()

        # Calculate expected result using numpy
        expected = np.einsum('ijk,jkl,lm->im', tensor1_data, tensor2_data, tensor3_data)

        print("CompTN result:")
        print(result)
        print("NumPy result:")
        print(expected)

        np.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_complex_intertwined_tensors(self):
        print("\n--- Test Complex Intertwined Tensors (5 tensors) ---")
        # Create 8 indices, each with dimension 2
        indices = [self.mm.Index(2) for _ in range(8)]

        # Create 5 tensors, each connected to three others
        tensor_data = [np.random.rand(2, 2, 2) for _ in range(5)]

        # Create tensors with intertwined indices
        tensors = [
            self.mm.Tensor(tensor_data[0], indices[0], indices[1], indices[2]),
            self.mm.Tensor(tensor_data[1], indices[0], indices[3], indices[4]),
            self.mm.Tensor(tensor_data[2], indices[1], indices[5], indices[6]),
            self.mm.Tensor(tensor_data[3], indices[2], indices[3], indices[7]),
            self.mm.Tensor(tensor_data[4], indices[4], indices[5], indices[7])
        ]

        # Contract all tensors
        self.mm.contract_multiple(*tensors)
        result = self.mm.run()

        # Calculate expected result using numpy einsum
        expected = np.einsum('abc,ade,bfg,cdh,efh->g',
                             *tensor_data)

        print("CompTN result:")
        print(result)
        print("NumPy result:")
        print(expected)

        np.testing.assert_allclose(result, expected, rtol=1e-5)


if __name__ == '__main__':
    unittest.main(verbosity=2)
