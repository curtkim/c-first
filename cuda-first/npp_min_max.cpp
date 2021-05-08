#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <npp.h>
//#include <nppi_statistics_functions.h>



typedef unsigned char byte;
byte* nppMaxBuffer;
float* max_d;
float* min_d;

int main(){

    int length = 5;
    float src_s[] = { -10, -5, 0, 3, 6 };
    float src2_s[] = { -55, -33, 0, 11, 22 };
    float *src_d, *src2_d;

    cudaMalloc(&src_d, sizeof(float) * length);
    cudaMalloc(&src2_d, sizeof(float)* length);
    cudaMalloc(&max_d, sizeof(float));
    cudaMalloc(&min_d, sizeof(float));

    int nBufferSize;
    nppsSumGetBufferSize_32f(length, &nBufferSize);
    cudaMalloc(&nppMaxBuffer, nBufferSize);

    cudaMemcpy(src_d, src_s, sizeof(float) * length, cudaMemcpyHostToDevice);
    cudaMemcpy(src2_d, src2_s, sizeof(float)* length, cudaMemcpyHostToDevice);


    nppsMax_32f(src_d, length, max_d, nppMaxBuffer);
    nppsMin_32f(src_d, length, min_d, nppMaxBuffer);

    float max_h = 0;
    float min_h = 0;

    cudaMemcpy(&max_h, max_d, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&min_h, min_d, sizeof(float), cudaMemcpyDeviceToHost);

    printf("nBufferSize : %d\n", nBufferSize);
    printf("src1 max_h : %f\n", max_h);
    printf("src1 min_h : %f\n", min_h);

    nppsMax_32f(src2_d, length, max_d, nppMaxBuffer);
    nppsMin_32f(src2_d, length, min_d, nppMaxBuffer);

    cudaMemcpy(&max_h, max_d, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&min_h, min_d, sizeof(float), cudaMemcpyDeviceToHost);

    printf("src2 max_h : %f\n", max_h);
    printf("src2 min_h : %f\n", min_h);

    return 0;
}