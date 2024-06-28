module {
  %0 = "tensor_network.index"() <{name = "i", size = 2 : i64}> : () -> !tensor_network.indexlabel
  %1 = "tensor_network.index"() <{name = "j", size = 3 : i64}> : () -> !tensor_network.indexlabel
  %2 = "tensor_network.index"() <{name = "k", size = 4 : i64}> : () -> !tensor_network.indexlabel
  %3 = "tensor_network.index"() <{name = "l", size = 5 : i64}> : () -> !tensor_network.indexlabel
  %4 = "tensor_network.tensor"(%0, %1) <{value = dense<[[0.28567017014427193, 0.25863560365448662, -0.31607917629601129], [0.13624746948030303, -1.6135630248961061, -1.8592507224929158]]> : tensor<2x3xf64>}> : (!tensor_network.indexlabel, !tensor_network.indexlabel) -> tensor<2x3xf64>
  %5 = "tensor_network.tensor"(%0, %2) <{value = dense<[[0.90034818767462799, 1.9128574885560501, -0.87138792457650172, -0.56747772475617841], [0.9508506476600872, -1.3142987535116972, 0.52998676176711201, 0.1645393142555161]]> : tensor<2x4xf64>}> : (!tensor_network.indexlabel, !tensor_network.indexlabel) -> tensor<2x4xf64>
  %6 = "tensor_network.contract"(%4, %5) : (tensor<2x3xf64>, tensor<2x4xf64>) -> tensor<2x3xf64>
}
