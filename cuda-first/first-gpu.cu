#include <iostream>
#include "first-gpu.hpp"

void printCudaVersion()
{
    int runtime_ver;
    cudaRuntimeGetVersion(&runtime_ver);
    std::cout << "CUDA Runtime version: " << runtime_ver << std::endl;

    int driver_ver;
    cudaDriverGetVersion(&driver_ver);
    std::cout << "CUDA Driver version: " << driver_ver << std::endl;
}

__global__ void print_from_gpu(void){
  printf("Hello World! from thread [%d, %d]\n", threadIdx.x, blockIdx.y);
}

void launch_print_from_gpu(){
  print_from_gpu<<<1,2>>>();
  cudaDeviceSynchronize();
}