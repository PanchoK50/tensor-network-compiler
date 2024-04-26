func.func @tensor_contract(%A: tensor<2x3xf32>, %B: tensor<3x4xf32>) -> tensor<2x4xf32> {
  %C = "tensor_network.contract"(%A, %B) : (tensor<2x3xf32>, tensor<3x4xf32>) -> tensor<2x4xf32>
  return %C : tensor<2x4xf32>
}
