#include <stdio.h>
#include "helper_cuda.h"
#include <thread>

#define N (1024*1024)
#define M (1000000)

__global__ void cudakernel(float *buf)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    buf[i] = 1.0f * i / N;
    for(int j = 0; j < M; j++)
        buf[i] = buf[i] * buf[i] - 0.25f;
}

void CUDART_CB myStreamCallback(cudaStream_t stream, cudaError_t status, void *data)
{
    // Check status of GPU after stream operations are done
    checkCudaErrors(status);
    float* fData = (float*)data;
    printf("callback thread(%u) data[0] = %f\n", std::this_thread::get_id(), fData[0]);
}

int main()
{
    float data[N];
    int count = 0;
    float *d_data;

    printf("main thread(%u)\n", std::this_thread::get_id());

    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreate(&stream));

    cudaMallocAsync(&d_data, N * sizeof(float), stream);
    cudakernel<<<N/256, 256, 0, stream>>>(d_data);
    cudaMemcpyAsync(data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaFreeAsync(d_data, stream);
    checkCudaErrors(cudaStreamAddCallback(stream, myStreamCallback, data, 0));

    printf("before destroy stream\n");
    cudaStreamDestroy(stream);
    printf("after destroy stream\n");

    getchar();
}