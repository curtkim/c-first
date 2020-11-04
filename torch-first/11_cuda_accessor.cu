#include <iostream>
#include <THC/THCAtomics.cuh>
#include <cmath>

using namespace std;
using namespace at;

const int blocksize = 256;
const int factor = 4;
const int arraysize = blocksize / factor;


template <typename T>
__global__ void addition_test_kernel(T * a, T * sum) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int idx = (tid) % arraysize;

  gpuAtomicAdd(&sum[idx], a[idx]);
}

template <typename T>
void test_atomic_add() {
  dim3 dimBlock(blocksize, 1);
  dim3 dimGrid(1, 1);

  T *a, *sum, *answer, *ad, *sumd;

  a = (T*)malloc(arraysize * sizeof(T));
  sum = (T*)malloc(arraysize * sizeof(T));
  answer = (T*)malloc(arraysize * sizeof(T));

  for (int i = 0; i < arraysize; ++i) {
    a[i] = 1;
    sum[i] = 0;
    answer[i] = factor;
  }

  cudaMalloc((void**)&ad, arraysize * sizeof(T));
  cudaMalloc((void**)&sumd, arraysize * sizeof(T));

  cudaMemcpy(ad, a, arraysize * sizeof(T), cudaMemcpyHostToDevice);
  cudaMemcpy(sumd, sum, arraysize * sizeof(T), cudaMemcpyHostToDevice);

  addition_test_kernel<<<dimGrid, dimBlock>>>(ad, sumd);

  cudaMemcpy(sum, sumd, arraysize * sizeof(T), cudaMemcpyDeviceToHost);

  for (int i = 0; i < arraysize; ++i) {
    cout << sum[i] << " " << answer[i] << typeid(T).name();
  }

  cudaFree(ad);
  cudaFree(sumd);
  free(a);
  free(sum);
  free(answer);
}

int main() {
  test_atomic_add<uint8_t>();
}