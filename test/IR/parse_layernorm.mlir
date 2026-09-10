// RUN: distilbert-opt %s | FileCheck %s

// CHECK-LABEL: func @test_layernorm_parse
func.func @test_layernorm_parse(%input: tensor<1x128x768xf32>, %weight: tensor<768xf32>, %bias: tensor<768xf32>) -> tensor<1x128x768xf32> {
  // CHECK: dbert.layernorm
  %0 = "dbert.layernorm"(%input, %weight, %bias) : (tensor<1x128x768xf32>, tensor<768xf32>, tensor<768xf32>) -> tensor<1x128x768xf32>
  func.return %0 : tensor<1x128x768xf32>
}
