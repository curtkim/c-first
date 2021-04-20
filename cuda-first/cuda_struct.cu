#include <stdlib.h>
#include <stdio.h>

struct Point
{
    float a, b;
};

__global__ void testKernel(Point *p)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    p[i].a = 1.1;
    p[i].b = 2.2;
}

int main(void)
{
    // set number of points
    int numPoints    = 16,
        pointSize    = sizeof(Point),
        numBytes     = numPoints * pointSize,
        gpuBlockSize = 4,
        gpuGridSize  = numPoints / gpuBlockSize;

    Point cpuPointArray[numPoints];
    Point* gpuPointArray;
    cudaMalloc((void**)&gpuPointArray, numBytes);

    // launch kernel
    testKernel<<<gpuGridSize,gpuBlockSize>>>(gpuPointArray);

    // retrieve the results
    cudaMemcpy(cpuPointArray, gpuPointArray, numBytes, cudaMemcpyDeviceToHost);
    printf("testKernel results:\n");
    for(int i = 0; i < numPoints; ++i)
    {
        printf("point.a: %f, point.b: %f\n",cpuPointArray[i].a, cpuPointArray[i].b);
    }

    // deallocate memory
    cudaFree(gpuPointArray);

    return 0;
}