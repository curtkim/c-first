/*
 * Copyright 2020 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
* This sample demonstrates stream ordered memory allocation on a GPU using
* cudaMallocAsync and cudaMemPool family of APIs.
*
* basicStreamOrderedAllocation(): demonstrates stream ordered allocation using
* cudaMallocAsync/cudaFreeAsync APIs with default settings.
*
* streamOrderedAllocationPostSync(): demonstrates if there's a synchronization in between allocations,
* then setting the release threshold on the pool will make sure the synchronize will not
* free memory back to the OS.
*/

// System includes
#include <stdio.h>
#include <assert.h>
#include <climits>

// CUDA runtime
#include <cuda_runtime.h>

// helper functions and utilities to work with CUDA
#include "common/helper_functions.h"
#include "common/helper_cuda.h"

#define MAX_ITER 20

/* Add two vectors on the GPU */
__global__ void vectorAddGPU(const float *a, const float *b, float *c, int N)
{
  int idx = blockIdx.x*blockDim.x + threadIdx.x;

  if (idx < N) {
    c[idx] = a[idx] + b[idx];
  }
}

int basicStreamOrderedAllocation(const int dev, const int nelem, const float *a, const float *b, float *c)
{
  float *d_a, *d_b, *d_c; // Device buffers
  float errorNorm, refNorm, ref, diff;
  size_t bytes = nelem * sizeof(float);

  cudaStream_t stream;
  printf("Starting basicStreamOrderedAllocation()\n");
  checkCudaErrors(cudaSetDevice(dev));
  checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

  checkCudaErrors(cudaMallocAsync(&d_a, bytes, stream));
  checkCudaErrors(cudaMallocAsync(&d_b, bytes, stream));
  checkCudaErrors(cudaMallocAsync(&d_c, bytes, stream));
  checkCudaErrors(cudaMemcpyAsync(d_a, a, bytes, cudaMemcpyHostToDevice, stream));
  checkCudaErrors(cudaMemcpyAsync(d_b, b, bytes, cudaMemcpyHostToDevice, stream));

  dim3 block(256);
  dim3 grid((unsigned int)ceil(nelem/(float)block.x));
  vectorAddGPU<<<grid, block, 0, stream>>>(d_a, d_b, d_c, nelem);

  checkCudaErrors(cudaFreeAsync(d_a, stream));
  checkCudaErrors(cudaFreeAsync(d_b, stream));
  checkCudaErrors(cudaMemcpyAsync(c, d_c, bytes, cudaMemcpyDeviceToHost, stream));
  checkCudaErrors(cudaFreeAsync(d_c, stream));
  checkCudaErrors(cudaStreamSynchronize(stream));

  /* Compare the results */
  printf("> Checking the results from vectorAddGPU() ...\n");
  errorNorm = 0.f;
  refNorm = 0.f;

  for (int n = 0; n < nelem; n++) {
    ref = a[n] + b[n];
    diff = c[n] - ref;
    errorNorm += diff*diff;
    refNorm += ref*ref;
  }

  errorNorm = (float)sqrt((double)errorNorm);
  refNorm = (float)sqrt((double)refNorm);
  if (errorNorm/refNorm < 1.e-6f)
    printf("basicStreamOrderedAllocation PASSED\n");

  checkCudaErrors(cudaStreamDestroy(stream));

  return errorNorm/refNorm < 1.e-6f ? EXIT_SUCCESS : EXIT_FAILURE;
}

// streamOrderedAllocationPostSync():
// demonstrates If the application wants the memory to persist in the pool beyond synchronization,
// then it sets the release threshold on the pool.
// This way, when the application reaches the "steady state",
// it is no longer allocating/freeing memory from the OS.
int streamOrderedAllocationPostSync(const int dev, const int nelem, const float *a, const float *b, float *c)
{
  float *d_a, *d_b, *d_c; // Device buffers
  float errorNorm, refNorm, ref, diff;
  size_t bytes = nelem * sizeof(float);

  cudaStream_t stream;
  cudaMemPool_t memPool;
  cudaEvent_t start, end;
  printf("Starting streamOrderedAllocationPostSync()\n");
  checkCudaErrors(cudaSetDevice(dev));
  checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&end));

  checkCudaErrors(cudaDeviceGetDefaultMemPool(&memPool, dev));
  uint64_t thresholdVal = ULONG_MAX;
  // set high release threshold on the default pool so that cudaFreeAsync will not actually release memory to the system.
  // By default, the release threshold for a memory pool is set to zero. This implies that the CUDA driver is
  // allowed to release a memory chunk back to the system as long as it does not contain any active suballocations.
  checkCudaErrors(cudaMemPoolSetAttribute(memPool, cudaMemPoolAttrReleaseThreshold, (void*)&thresholdVal));

  // Record the start event
  checkCudaErrors(cudaEventRecord(start, stream));
  for (int i = 0; i < MAX_ITER; i++)
  {
    checkCudaErrors(cudaMallocAsync(&d_a, bytes, stream));
    checkCudaErrors(cudaMallocAsync(&d_b, bytes, stream));
    checkCudaErrors(cudaMallocAsync(&d_c, bytes, stream));
    checkCudaErrors(cudaMemcpyAsync(d_a, a, bytes, cudaMemcpyHostToDevice, stream));
    checkCudaErrors(cudaMemcpyAsync(d_b, b, bytes, cudaMemcpyHostToDevice, stream));

    dim3 block(256);
    dim3 grid((unsigned int)ceil(nelem/(float)block.x));
    vectorAddGPU<<<grid, block, 0, stream>>>(d_a, d_b, d_c, nelem);

    checkCudaErrors(cudaFreeAsync(d_a, stream));
    checkCudaErrors(cudaFreeAsync(d_b, stream));
    checkCudaErrors(cudaMemcpyAsync(c, d_c, bytes, cudaMemcpyDeviceToHost, stream));
    checkCudaErrors(cudaFreeAsync(d_c, stream));
    checkCudaErrors(cudaStreamSynchronize(stream));
  }
  checkCudaErrors(cudaEventRecord(end, stream));
  // Wait for the end event to complete
  checkCudaErrors(cudaEventSynchronize(end));

  float msecTotal = 0.0f;
  checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, end));
  printf("Total elapsed time = %f ms over %d iterations\n", msecTotal, MAX_ITER);

  /* Compare the results */
  printf("> Checking the results from vectorAddGPU() ...\n");
  errorNorm = 0.f;
  refNorm = 0.f;

  for (int n = 0; n < nelem; n++) {
    ref = a[n] + b[n];
    diff = c[n] - ref;
    errorNorm += diff*diff;
    refNorm += ref*ref;
  }

  errorNorm = (float)sqrt((double)errorNorm);
  refNorm = (float)sqrt((double)refNorm);
  if (errorNorm/refNorm < 1.e-6f)
    printf("streamOrderedAllocationPostSync PASSED\n");

  checkCudaErrors(cudaStreamDestroy(stream));

  return errorNorm/refNorm < 1.e-6f ? EXIT_SUCCESS : EXIT_FAILURE;
}


int main(int argc, char **argv)
{
  int nelem;
  int dev = 0; // use default device 0
  size_t bytes;
  float *a, *b, *c; // Host

  if (checkCmdLineFlag(argc, (const char **)argv, "help"))
  {
    printf("Usage:  streamOrderedAllocation [OPTION]\n\n");
    printf("Options:\n");
    printf("  --device=[device #]  Specify the device to be used\n");
    return EXIT_SUCCESS;
  }

  dev = findCudaDevice(argc, (const char **)argv);

  int isMemPoolSupported = 0;
  checkCudaErrors(cudaDeviceGetAttribute(&isMemPoolSupported, cudaDevAttrMemoryPoolsSupported, dev));
  if (!isMemPoolSupported)
  {
    printf("Waiving execution as device does not support Memory Pools\n");
    exit(EXIT_WAIVED);
  }

  // Allocate CPU memory.
  nelem = 1048576;
  bytes = nelem*sizeof(float);

  a = (float*) malloc(bytes);
  b = (float*) malloc(bytes);
  c = (float*) malloc(bytes);
  /* Initialize the vectors. */
  for (int n = 0; n < nelem; n++)
  {
    a[n] = rand() / (float)RAND_MAX;
    b[n] = rand() / (float)RAND_MAX;
  }

  int ret1 = basicStreamOrderedAllocation(dev, nelem, a, b, c);
  int ret2 = streamOrderedAllocationPostSync(dev, nelem, a, b, c);

  /* Memory clean up */
  free(a);
  free(b);
  free(c);

  return ((ret1 == EXIT_SUCCESS && ret2 == EXIT_SUCCESS) ? EXIT_SUCCESS : EXIT_FAILURE);
}