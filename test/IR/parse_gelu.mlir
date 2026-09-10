// RUN: distilbert-opt %s | FileCheck %s

// CHECK-LABEL: func @test_gelu_parse
func.func @test_gelu_parse(%input: tensor<1x128x768xf32>) -> tensor<1x128x768xf32> {
  // CHECK: dbert.gelu
  %0 = "dbert.gelu"(%input) : (tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
  func.return %0 : tensor<1x128x768xf32>
}
