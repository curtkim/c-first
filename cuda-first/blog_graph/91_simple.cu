#include <chrono>
#include <iostream>
#include <iomanip>

#define N 500000 // tuned such that kernel takes a few microseconds

__global__ void shortKernel(float * out_d, float * in_d){
  int idx=blockIdx.x*blockDim.x+threadIdx.x;
  if(idx<N) out_d[idx]=1.23*in_d[idx];
}

#define NSTEP 1000
#define NKERNEL 20

int main()
{
  std::chrono::system_clock::time_point now = std::chrono::system_clock::now();

  cudaStream_t stream = 0;

  float *data_in;
  cudaMalloc(&data_in, N * sizeof(float));
  float *data_out;
  cudaMalloc(&data_out, N * sizeof(float));

  // start CPU wallclock timer
  for(int istep=0; istep<NSTEP; istep++){
    for(int ikrnl=0; ikrnl<NKERNEL; ikrnl++){
      shortKernel<<<N/256, 256, 0, stream>>>(data_out, data_in);
    }
    cudaStreamSynchronize(0);
  }

  std::chrono::duration<double> d = std::chrono::duration<double>(now.time_since_epoch());
  std::cout << std::setprecision (std::numeric_limits<double>::digits10 + 1)
            << d.count() / (NSTEP*NKERNEL) << std::endl;
}
