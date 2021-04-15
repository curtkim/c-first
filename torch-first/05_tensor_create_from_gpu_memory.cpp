#include <torch/torch.h>
#include <iostream>
#include <cuda_runtime_api.h>
#include <cuda.h>

const int N = 10;

void use_another_tensor() {
  auto options =
    torch::TensorOptions()
      .dtype(torch::kFloat32)
      .device(torch::kCUDA, 0);

  float blob[] = {0,1,2,3,4,5,6,7,8,9};
  auto a = torch::from_blob(&blob, {N}, torch::kFloat32);
  std::cout << "torch::tensor sizeof " << sizeof(a) << std::endl;
  auto b = a.to(torch::kCUDA);

  torch::Tensor cudaTest = torch::from_blob(b.data_ptr(), {N}, options);
  std::cout << cudaTest << std::endl;
  std::cout << cudaTest.sum() << std::endl;
}

void use_direct_cuda_memory() {
  float data[N] = {0,1,2,3,4,5,6,7,8,9};

  void* d_data;
  cudaSetDevice(0);
  cudaMalloc(&d_data, N * sizeof(float));
  cudaMemcpy(d_data, data, N * sizeof(float), cudaMemcpyHostToDevice);

  auto options =
    torch::TensorOptions()
      .dtype(torch::kFloat32)
      .device(torch::kCUDA, 0);

  // 에러가 발생한다.
  // Specified device cuda:0 does not match device of data cuda:-2
  torch::Tensor cudaTest = torch::from_blob(&d_data, {N}, options);
  std::cout << cudaTest << std::endl;
  std::cout << cudaTest.sum() << std::endl;

  cudaFree(d_data);

}

int main() {

  int cudaCount;
  cudaGetDeviceCount(&cudaCount);
  std::cout << "cudaGetDeviceCount " << cudaCount << std::endl;

  int currentDevice;
  cudaGetDevice(&currentDevice);
  std::cout << "cudaGetDevice " << currentDevice << std::endl;

  use_another_tensor();
  use_direct_cuda_memory();
}