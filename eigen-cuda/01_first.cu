#include <cuda.h>
#include <cuda_runtime_api.h>

#include <iostream>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/src/Tensor/TensorGpuHipCudaDefines.h>


template<typename T>
struct redux {
    EIGEN_DEVICE_FUNC void operator()(int i, const typename T::Scalar* in, typename T::Scalar* out) const
    {
        using namespace Eigen;
        int N = 10;
        T x1(in+i);
        out[i*N+0] = x1.minCoeff();
        out[i*N+1] = x1.maxCoeff();
        out[i*N+2] = x1.sum();
        out[i*N+3] = x1.prod();
        out[i*N+4] = x1.matrix().squaredNorm();
        out[i*N+5] = x1.matrix().norm();
        out[i*N+6] = x1.colwise().sum().maxCoeff();
        out[i*N+7] = x1.rowwise().maxCoeff().sum();
        out[i*N+8] = x1.matrix().colwise().squaredNorm().sum();
    }
};


template<typename Kernel, typename Input, typename Output>
__global__ EIGEN_HIP_LAUNCH_BOUNDS_1024
void run_on_gpu_meta_kernel(const Kernel ker, int n, const Input* in, Output* out)
{
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if(i<n) {
        ker(i, in, out);
    }
}


template<typename Kernel, typename Input, typename Output>
void run_on_gpu(const Kernel& ker, int n, const Input& in, Output& out)
{
    typename Input::Scalar*  d_in;
    typename Output::Scalar* d_out;
    std::ptrdiff_t in_bytes  = in.size()  * sizeof(typename Input::Scalar);
    std::ptrdiff_t out_bytes = out.size() * sizeof(typename Output::Scalar);

    gpuMalloc((void**)(&d_in),  in_bytes);
    gpuMalloc((void**)(&d_out), out_bytes);

    gpuMemcpy(d_in,  in.data(),  in_bytes,  gpuMemcpyHostToDevice);
    gpuMemcpy(d_out, out.data(), out_bytes, gpuMemcpyHostToDevice);

    // Simple and non-optimal 1D mapping assuming n is not too large
    // That's only for unit testing!
    dim3 Blocks(128);
    dim3 Grids( (n+int(Blocks.x)-1)/int(Blocks.x) );

    gpuDeviceSynchronize();

    run_on_gpu_meta_kernel<<<Grids,Blocks>>>(ker, n, d_in, d_out);

    // Pre-launch errors.
    gpuError_t err = gpuGetLastError();
    if (err != gpuSuccess) {
        printf("%s: %s\n", gpuGetErrorName(err), gpuGetErrorString(err));
        gpu_assert(false);
    }

    // Kernel execution errors.
    err = gpuDeviceSynchronize();
    if (err != gpuSuccess) {
        printf("%s: %s\n", gpuGetErrorName(err), gpuGetErrorString(err));
        gpu_assert(false);
    }


    // check inputs have not been modified
    gpuMemcpy(const_cast<typename Input::Scalar*>(in.data()),  d_in,  in_bytes,  gpuMemcpyDeviceToHost);
    gpuMemcpy(out.data(), d_out, out_bytes, gpuMemcpyDeviceToHost);

    gpuFree(d_in);
    gpuFree(d_out);
}

int main() {
    Eigen::VectorXf in, out;
    int nthreads = 100;

    int data_size = nthreads * 512;
    in.setRandom(data_size);
    out.setConstant(data_size, -1);

    run_on_gpu(redux<Eigen::Array4f>(), nthreads, in, out);
    return 0;
}
