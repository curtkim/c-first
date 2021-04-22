#include <ATen/ATen.h>
#include <iostream>
#include <assert.h>
#include <torch/torch.h>

#include <ATen/cuda/CUDAContext.h>

using namespace std;

int main() {

  cout << "CUDA: " << (torch::cuda::is_available() ? "Yes" : "No") << endl;
  cout << "cuDNN: " << (torch::cuda::cudnn_is_available() ? "Yes" : "No") << endl;

  at::Tensor a = at::ones({2, 2}, at::kInt);
  at::Tensor b = at::randn({2, 2}); // float type
  auto c = a + b.to(at::kInt);

  cout << a << endl;
  cout << b << endl;
  cout << c << endl;

  cout << a.sum(at::kInt) << endl;

  at::Allocator * allocator = at::cuda::getCUDADeviceAllocator();

  return 0;
}

