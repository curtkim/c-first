// https://github.com/NVIDIA/thrust/tree/main/examples/cpp_integration
//
// Note that device_vector only appears in the .cu file while host_vector appears in both.
// This relects the fact that
// algorithms on device vectors are only available
// when the contents of the program are located in a .cu file and compiled with the nvcc compiler.
//
// $ nvcc -O2 -c device.cu
// $ g++  -O2 -c host.cpp   -I/usr/local/cuda/include/
// $ g++ -o tester device.o host.o -L/usr/local/cuda/lib64 -lcudart

#include <thrust/host_vector.h>
#include <thrust/random.h>
#include <thrust/generate.h>
#include <thrust/sort.h>
#include <cstdlib>
#include <iostream>
#include <iterator>

// defines the function prototype
#include "99_device.h"

int main(void)
{
    // generate 20 random numbers on the host
    thrust::host_vector<int> h_vec(20);
    thrust::default_random_engine rng;
    thrust::generate(h_vec.begin(), h_vec.end(), rng);

    // interface to CUDA code
    sort_on_device(h_vec);

    // print sorted array
    thrust::copy(h_vec.begin(), h_vec.end(), std::ostream_iterator<int>(std::cout, "\n"));

    return 0;
}