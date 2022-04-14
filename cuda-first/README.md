## raw compile & link by nvcc and c++

    nvcc -c first-gpu.cu -o build/first-gpu.o
    c++ -c first.cc -o build/first.o
    c++ build/first.o build/first-gpu.o -o build/first -L/usr/local/cuda/targets/x86_64-linux/lib -lcudadevrt -lcudart_static -lpthread -ldl
    # 링크명령은 cmake-build-debug/CMakeFiles/first.dir/link.txt에서 참조함.


## CMake options in clion

    -DCMAKE_CUDA_COMPILER=/usr/local/cuda-10.0/bin/nvcc


## nvtx

    nsys profile -o baseline.qdstrm --force-overwrite=true ./nvtx_first
    nsys-ui 



## Reference
https://www.jetbrains.com/help/clion/cuda-projects.html
https://github.com/socal-ucr/CUDA-Intro
https://github.com/sol-prog/cuda_cublas_curand_thrust
https://docs.nvidia.com/cuda/cuda-samples/index.html#simple
