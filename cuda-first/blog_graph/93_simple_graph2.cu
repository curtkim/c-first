// https://github.com/plops/cl-cpp-generator2/blob/2e2080e6e094f5d57ec518d1cc0b9b2d2a57e219/example/24_cuda_graph_launch/source/globals.h
#include <array>
#include <iomanip>
#include <iostream>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <thread>

#include <cuda.h>
#include <cuda_runtime.h>

enum { N = 500000, NSTEP = 1000, NKERNEL = 20 };
using namespace std::chrono_literals;

__global__ void shortKernel(float *out, float *in) {
  auto idx = ((((blockIdx.x) * (blockDim.x))) + (threadIdx.x));
  if ((idx) < (N)) {
    out[idx] = ((in[idx]) * ((1.230f)));
  };
}
void init_input(float *a, size_t size) {
  for (auto i = 0; (i) < (size); (i) += (1)) {
    a[i] = (((1.0f)) * (i));
  }
}


int main(int argc, char const *const *const argv) {

  cudaStream_t stream;
  auto blocks = 512;
  auto threads = 512;
  if (!((cudaSuccess) == (cudaStreamCreate(&stream)))) {
    throw std::runtime_error("cudaStreamCreate(&stream)");
  };

  float *in;
  float *out;

  if (!((cudaSuccess) == (cudaMallocManaged(&in, ((N) * (sizeof(float))))))) {
    throw std::runtime_error("cudaMallocManaged(&in, ((N)*(sizeof(float))))");
  };
  if (!((cudaSuccess) == (cudaMallocManaged(&out, ((N) * (sizeof(float))))))) {
    throw std::runtime_error("cudaMallocManaged(&out, ((N)*(sizeof(float))))");
  };

  init_input(in, N);
  auto graph_created = false;

  cudaGraph_t graph;
  cudaGraphExec_t instance;

  std::chrono::system_clock::time_point now = std::chrono::system_clock::now();

  for (auto istep = 0; istep < NSTEP; istep ++) {
    if (!graph_created) {
      cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
      for (auto ik = 0; ik < NKERNEL; ik++) {
        shortKernel<<<blocks, threads, 0, stream>>>(out, in);
      }
      cudaStreamEndCapture(stream, &graph);
      cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0);
      graph_created = true;
    };
    cudaGraphLaunch(instance, stream);
    cudaStreamSynchronize(stream);
  };
  if (!((cudaSuccess) == (cudaFree(in)))) {
    throw std::runtime_error("cudaFree(in)");
  };
  if (!((cudaSuccess) == (cudaFree(out)))) {
    throw std::runtime_error("cudaFree(out)");
  };

  std::chrono::duration<double> d = std::chrono::duration<double>(now.time_since_epoch());
  std::cout << std::setprecision (std::numeric_limits<double>::digits10 + 1)
            << d.count() / (NSTEP*NKERNEL) << std::endl;

  return 0;
};