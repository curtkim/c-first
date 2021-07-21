## idea 1

    auto kernel = [](stream& stream, a, b){
        
    }
    co_await cuda.launch<a,b,stream,device>(kernel, src, dst);

    ---

    stream_t stream
    cudaCreateStream(&stream)
    cudaMallocAsync(&d_src, SIZE, stream);
    cudaMallocAsync(&d_dst, SIZE, stream);

    cudaMemcpyAsync(d_src, src, SIZE, cudaMemcpyHostToDevice, stream); 
    kernel(d_src, d_dst)
    cudaMemcpyAsync(dst, d_dst, SIZE, cudaMemcpyDeviceToHost, stream);

    cudaFreeAsync(d_src, stream);
    cudaFreeAsync(d_dst, stream);
    cudaStreamAddCallback()
    cudaDestroyStream(stream)


## idea 2

    // 내부적으로 하나의 stream을 쓴다.
    Task task1() { 
        when_all(
            co_await cuda.launch<a,b>(kernel1, src1, dst1),
            co_await cuda.launch<a,b>(kernel2, src2, dst2)
        , [](){
          co_await cuda.launch<a,b>(kernel3, dst1, dst2)
        })
    }