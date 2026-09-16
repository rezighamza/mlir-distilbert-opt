// RUN: distilbert-opt %s --dbert-fuse-attention | FileCheck %s

// CHECK-LABEL: func @test_fuse_attention
func.func @test_fuse_attention(%query: tensor<1x128x768xf32>, %key: tensor<1x128x768xf32>, %value: tensor<1x128x768xf32>) -> tensor<1x128x768xf32> {
  // CHECK-NOT: dbert.matmul
  // CHECK-NOT: dbert.softmax
  // CHECK: dbert.mha
  
  // 1. MatMul(Query, Key)
  %score = "dbert.matmul"(%query, %key) : (tensor<1x128x768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x128xf32>
  
  // 2. Softmax(Score)
  %probs = "dbert.softmax"(%score) : (tensor<1x128x128xf32>) -> tensor<1x128x128xf32>
  
  // 3. MatMul(Probs, Value)
  %out = "dbert.matmul"(%probs, %value) : (tensor<1x128x128xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
  
  func.return %out : tensor<1x128x768xf32>
}
