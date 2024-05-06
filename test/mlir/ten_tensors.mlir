module {
  %0 = "tensor_network.tensor"() <{value = dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf64>}> : () -> tensor<2x2xf64>
  %1 = "tensor_network.tensor"() <{value = dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf64>}> : () -> tensor<2x2xf64>
  %2 = "tensor_network.tensor"() <{value = dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf64>}> : () -> tensor<2x2xf64>
  %3 = "tensor_network.tensor"() <{value = dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf64>}> : () -> tensor<2x2xf64>
  %4 = "tensor_network.tensor"() <{value = dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf64>}> : () -> tensor<2x2xf64>
  %5 = "tensor_network.tensor"() <{value = dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf64>}> : () -> tensor<2x2xf64>
  %6 = "tensor_network.tensor"() <{value = dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf64>}> : () -> tensor<2x2xf64>
  %7 = "tensor_network.tensor"() <{value = dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf64>}> : () -> tensor<2x2xf64>
  %8 = "tensor_network.tensor"() <{value = dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf64>}> : () -> tensor<2x2xf64>
  %9 = "tensor_network.tensor"() <{value = dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf64>}> : () -> tensor<2x2xf64>
  %10 = tensor_network.contraction_edge {contracted_indices = [0, 1]} %0 : tensor<2x2xf64> with %1 : tensor<2x2xf64> to tensor<2x2xf64>
  %11 = tensor_network.contraction_edge {contracted_indices = [0, 1]} %1 : tensor<2x2xf64> with %2 : tensor<2x2xf64> to tensor<2x2xf64>
  %12 = tensor_network.contraction_edge {contracted_indices = [0, 1]} %2 : tensor<2x2xf64> with %3 : tensor<2x2xf64> to tensor<2x2xf64>
  %13 = tensor_network.contraction_edge {contracted_indices = [0, 1]} %3 : tensor<2x2xf64> with %4 : tensor<2x2xf64> to tensor<2x2xf64>
  %14 = tensor_network.contraction_edge {contracted_indices = [0, 1]} %4 : tensor<2x2xf64> with %5 : tensor<2x2xf64> to tensor<2x2xf64>
  %15 = tensor_network.contraction_edge {contracted_indices = [0, 1]} %5 : tensor<2x2xf64> with %6 : tensor<2x2xf64> to tensor<2x2xf64>
  %16 = tensor_network.contraction_edge {contracted_indices = [0, 1]} %6 : tensor<2x2xf64> with %7 : tensor<2x2xf64> to tensor<2x2xf64>
  %17 = tensor_network.contraction_edge {contracted_indices = [0, 1]} %7 : tensor<2x2xf64> with %8 : tensor<2x2xf64> to tensor<2x2xf64>
  %18 = tensor_network.contraction_edge {contracted_indices = [0, 1]} %8 : tensor<2x2xf64> with %9 : tensor<2x2xf64> to tensor<2x2xf64>
}
