extern "C" __global__ void kernelFunction(int *input)
{
    input[threadIdx.x] = 32 - threadIdx.x;
}