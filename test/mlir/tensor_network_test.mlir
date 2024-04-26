func.func @main() -> tensor<f64> {
    %0 = "tensor_network.constant_tensor"() {value = dense<[1.0]> : tensor<1xf64>} : () -> tensor<f64>
    %1 = "tensor_network.constant_tensor"() {value = dense<[1.0, 2.0, 3.0]> : tensor<3xf64>} : () -> tensor<f64>
    %2 = "tensor_network.constant_tensor"() {value = dense<[1.0, 2.0, 4.0]> : tensor<3xf64>} : () -> tensor<f64>
    return %0 : tensor<f64>
}