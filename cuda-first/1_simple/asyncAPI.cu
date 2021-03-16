//
// This sample illustrates the usage of CUDA events for both GPU timing and
// overlapping CPU and GPU execution.  Events are inserted into a stream
// of CUDA calls.  Since CUDA stream calls are asynchronous, the CPU can
// perform computations while GPU is executing (including DMA memcopies
// between the host and device).  CPU can query CUDA events to determine
// whether GPU has completed tasks.
//

// includes, system
#include <stdio.h>

// includes CUDA Runtime
#include <cuda_runtime.h>

// includes, project
#include <helper_cuda.h>
#include <helper_functions.h> // helper utility functions

__global__ void increment_kernel(int *g_data, int inc_value)
{
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  g_data[idx] = g_data[idx] + inc_value;
}

bool correct_output(int *data, const int n, const int x)
{
  for (int i = 0; i < n; i++)
    if (data[i] != x)
    {
      printf("Error! data[%d] = %d, ref = %d\n", i, data[i], x);
      return false;
    }

  return true;
}

int main(int argc, char *argv[])
{
  int devID;
  cudaDeviceProp deviceProps;

  printf("[%s] - Starting...\n", argv[0]);

  // This will pick the best possible CUDA capable device
  devID = findCudaDevice(argc, (const char **)argv);

  // get device name
  checkCudaErrors(cudaGetDeviceProperties(&deviceProps, devID));
  printf("CUDA device [%s]\n", deviceProps.name);


  const int N = 1024 * 1024;
  const int N_BYTES = N * sizeof(int);
  const int VALUE = 26;


  // allocate host memory
  int* host_arr;
  checkCudaErrors(cudaMallocHost((void **)&host_arr, N_BYTES));
  memset(host_arr, 0, N_BYTES);

  // allocate device memory
  int* device_arr;
  checkCudaErrors(cudaMalloc((void **)&device_arr, N_BYTES));
  checkCudaErrors(cudaMemset(device_arr, 255, N_BYTES));

  // create cuda event handles
  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  StopWatchInterface *timer = NULL;
  sdkCreateTimer(&timer);
  sdkResetTimer(&timer);


  checkCudaErrors(cudaDeviceSynchronize());
  float gpu_time = 0.0f;

  // asynchronously issue work to the GPU (all to stream 0)
  sdkStartTimer(&timer);
  cudaEventRecord(start, 0);
  cudaMemcpyAsync(device_arr, host_arr, N_BYTES, cudaMemcpyHostToDevice, 0);

  // set kernel launch configuration
  dim3 threads = dim3(512, 1);
  dim3 blocks  = dim3(N / threads.x, 1);
  increment_kernel<<<blocks, threads, 0, 0>>>(device_arr, VALUE);

  cudaMemcpyAsync(host_arr, device_arr, N_BYTES, cudaMemcpyDeviceToHost, 0);
  cudaEventRecord(stop, 0);
  sdkStopTimer(&timer);


  // have CPU do some work while waiting for stage 1 to finish
  unsigned long int counter=0;
  while (cudaEventQuery(stop) == cudaErrorNotReady)
  {
    counter++;
  }

  checkCudaErrors(cudaEventElapsedTime(&gpu_time, start, stop));

  // print the cpu and gpu times
  printf("time spent executing by the GPU: %.2f\n", gpu_time);
  printf("time spent by CPU in CUDA calls: %.2f\n", sdkGetTimerValue(&timer));
  printf("CPU executed %lu iterations while waiting for GPU to finish\n", counter);

  // check the output for correctness
  bool bFinalResults = correct_output(host_arr, N, VALUE);

  // release resources
  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));
  checkCudaErrors(cudaFreeHost(host_arr));
  checkCudaErrors(cudaFree(device_arr));

  exit(bFinalResults ? EXIT_SUCCESS : EXIT_FAILURE);
}