module {
  func.func @main() {
    %0 = "tensor_network.index"() <{name = "i", size = 2 : i64}> : () -> !tensor_network.indexlabel
    %1 = "tensor_network.index"() <{name = "j", size = 3 : i64}> : () -> !tensor_network.indexlabel
    %2 = "tensor_network.index"() <{name = "k", size = 4 : i64}> : () -> !tensor_network.indexlabel
    %3 = "tensor_network.index"() <{name = "l", size = 5 : i64}> : () -> !tensor_network.indexlabel
    %4 = "tensor_network.tensor"(%0, %1) <{value = dense<[[-0.23151269951472447, -0.4189003132480254, 0.68527688667295594], [0.67783949520819831, 0.75630516152279481, -0.18050263465206412]]> : tensor<2x3xf64>}> : (!tensor_network.indexlabel, !tensor_network.indexlabel) -> tensor<2x3xf64>
    %5 = "tensor_network.tensor"(%0, %2) <{value = dense<[[-0.36175536457560054, -1.0606623481849045, 1.0084961361186553, 0.68254268674267005], [0.66355119013781494, 0.016191642708408133, 2.5593254357214534, -0.17375587221679509]]> : tensor<2x4xf64>}> : (!tensor_network.indexlabel, !tensor_network.indexlabel) -> tensor<2x4xf64>
    %6 = "tensor_network.contract"(%4, %5) : (tensor<2x3xf64>, tensor<2x4xf64>) -> tensor<f64>
    return
  }
}
