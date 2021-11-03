#include <chrono>
#include <iostream>
#include <iomanip>

enum { N = 500000, NSTEP = 1000, NKERNEL = 20 };

__global__ void shortKernel(float * out_d, float * in_d){
  int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<N) out_d[idx]=1.23*in_d[idx];
}


int main()
{
  cudaStream_t stream;
  auto blocks = 512;
  auto threads = 512;
  if (!((cudaSuccess) == (cudaStreamCreate(&stream)))) {
    throw std::runtime_error("cudaStreamCreate(&stream)");
  };


  float *data_in;
  cudaMalloc(&data_in, N * sizeof(float));
  float *data_out;
  cudaMalloc(&data_out, N * sizeof(float));

  std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
  // start CPU wallclock timer
  for(int istep=0; istep<NSTEP; istep++){
    for(int ikrnl=0; ikrnl<NKERNEL; ikrnl++){
      shortKernel<<<blocks, threads, 0, stream>>>(data_out, data_in);
      cudaStreamSynchronize(0);
    }
  }

  std::chrono::duration<double> d = std::chrono::duration<double>(now.time_since_epoch());
  std::cout << std::setprecision (std::numeric_limits<double>::digits10 + 1)
            << d.count() / (NSTEP*NKERNEL) << std::endl;
}
