#include <torch/torch.h>

__global__ void cudakernel(int* buf)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    buf[i] = buf[i]+1;
}

int main() {
    auto a = torch::zeros({16, 16}, at::device(at::kCUDA).dtype(at::kInt));
    std::cout << a << "\n";
    cudakernel<<<16, 16>>>((int*)a.data_ptr());
    std::cout << a << "\n";
}