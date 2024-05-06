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
  %10 = "tensor_network.contraction_edge"(%0, %1) : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
  %11 = "tensor_network.contraction_edge"(%1, %2) : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
  %12 = "tensor_network.contraction_edge"(%2, %3) : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
  %13 = "tensor_network.contraction_edge"(%3, %4) : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
  %14 = "tensor_network.contraction_edge"(%4, %5) : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
  %15 = "tensor_network.contraction_edge"(%5, %6) : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
  %16 = "tensor_network.contraction_edge"(%6, %7) : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
  %17 = "tensor_network.contraction_edge"(%7, %8) : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
  %18 = "tensor_network.contraction_edge"(%8, %9) : (tensor<2x2xf64>, tensor<2x2xf64>) -> tensor<2x2xf64>
}
