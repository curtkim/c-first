#include <iostream>

#include "gpu.hpp"

int main()
{
    std::cout << "Hello, world!" << std::endl;

    std::cout << "CUDA: On" << std::endl;
    printCudaVersion();

    launch_print_from_gpu();
    return 0;
}