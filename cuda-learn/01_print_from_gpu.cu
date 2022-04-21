#include <iostream>

__global__ void print_from_gpu(void) {
    printf("Hello world from thread [%d, %d]\n", blockIdx.x, threadIdx.x);
}

int main() {
  print_from_gpu<<<1, 2>>>();
  cudaDeviceSynchronize();
  return 0;
}
