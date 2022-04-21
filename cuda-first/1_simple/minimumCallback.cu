#include <stdio.h>
#include "helper_cuda.h"
#include <thread>

#define N (1024*1024)
#define M (1000000)

__global__ void myKernel(float *buf)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    buf[i] = 1.0f * i;// / N;
//    for(int j = 0; j < M; j++)
//        buf[i] = buf[i] * buf[i] - 0.25f;
}

// CUDART_CB는 필수는 아닌 것 같다.
// stream이 이 시점에는 Destroy 되었는데.. 괜찮은가?
void CUDART_CB myCallback(cudaStream_t stream, cudaError_t status, void *data)
{
    // Check status of GPU after stream operations are done
    checkCudaErrors(status);
    float* fData = (float*)data;
    printf("callback thread(%u) data[0] = %f\n", std::this_thread::get_id(), fData[1]);
}

int main()
{
    float data[N];
    float *d_data;

    printf("main thread(%u)\n", std::this_thread::get_id());

    cudaStream_t stream;
    checkCudaErrors(cudaStreamCreate(&stream));

    cudaMallocAsync(&d_data, N * sizeof(float), stream);
    myKernel<<<N / 256, 256, 0, stream>>>(d_data);
    cudaMemcpyAsync(data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaFreeAsync(d_data, stream);
    checkCudaErrors(cudaStreamAddCallback(stream, myCallback, data, 0));

    printf("before destroy stream\n");
    cudaStreamDestroy(stream);
    printf("after destroy stream\n");

    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    //getchar();
}

// main thread(2069430272)
// before destroy stream
// after destroy stream
// callback thread(1892225024) data[0] = -0.207107

// main과 callback 스레드가 다르다.