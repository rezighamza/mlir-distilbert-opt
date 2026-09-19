// RUN: distilbert-opt %s --dbert-quantize-int8 | FileCheck %s

// CHECK-LABEL: func @test_quantize
func.func @test_quantize(%lhs: tensor<128x768xf32>, %rhs: tensor<768x128xf32>) -> tensor<128x128xf32> {
  // CHECK: dbert.quantize
  // CHECK-SAME: scale(0.07874
  // CHECK: dbert.quantize
  // CHECK-SAME: scale(0.07874
  // CHECK: dbert.matmul{{.*}}tensor<128x768xi8>
  // CHECK: dbert.dequantize
  // CHECK-SAME: scale(0.07874
  
  // Here the frontend has supplied calibration stats: min=-5.0, max=10.0
  // max_abs = 10.0. Scale = 10.0 / 127 = 0.078740...
  %0 = "dbert.matmul"(%lhs, %rhs) {calib_min = -5.0 : f32, calib_max = 10.0 : f32} : (tensor<128x768xf32>, tensor<768x128xf32>) -> tensor<128x128xf32>
  func.return %0 : tensor<128x128xf32>
}
