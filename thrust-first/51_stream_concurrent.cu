#include <thrust/for_each.h>
#include <thrust/system/cuda/execution_policy.h>

struct printf_functor
{
    cudaStream_t s;
    printf_functor(cudaStream_t s) : s(s) {}

    __host__ __device__ void operator()(int)
    {
        printf("Hello, world from stream %p\n", static_cast<void*>(s));
    }
};

int main()
{
    // create two CUDA streams
    cudaStream_t s1, s2;
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);

    thrust::counting_iterator<int> iter(0);

    // execute for_each on two different streams
    thrust::for_each(thrust::cuda::par.on(s1), iter, iter + 1, printf_functor(s1));
    thrust::for_each(thrust::cuda::par.on(s2), iter, iter + 1, printf_functor(s2));

    // synchronize with both streams
    cudaStreamSynchronize(s1);
    cudaStreamSynchronize(s2);

    // destroy streams
    cudaStreamDestroy(s1);
    cudaStreamDestroy(s2);

    return 0;
}