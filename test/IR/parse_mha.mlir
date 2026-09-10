// RUN: distilbert-opt %s | FileCheck %s

// CHECK-LABEL: func @test_mha_parse
func.func @test_mha_parse(%query: tensor<1x128x768xf32>, %key: tensor<1x128x768xf32>, %value: tensor<1x128x768xf32>) -> tensor<1x128x768xf32> {
  // CHECK: dbert.mha
  %0 = "dbert.mha"(%query, %key, %value) : (tensor<1x128x768xf32>, tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
  func.return %0 : tensor<1x128x768xf32>
}
