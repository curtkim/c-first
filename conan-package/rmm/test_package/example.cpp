#include <assert.h>

#include <rmm/cuda_stream_pool.hpp>
#include <rmm/detail/error.hpp>
#include <rmm/device_uvector.hpp>

#include <cuda_runtime_api.h>


int main(int argc, char ** argv) {
    rmm::cuda_stream_pool pool{};

    auto const stream_a = pool.get_stream();
    auto const stream_b = pool.get_stream();
    assert(stream_a == stream_b);

    assert(!stream_a.is_default());
    assert(!stream_a.is_per_thread_default());

    return 0;
}