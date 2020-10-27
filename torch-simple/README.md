## Build

    make VERBOSE=1
    
    # compile 
    /usr/bin/c++  -isystem /home/curt/.conan/data/torch/1.6.0/curt/prebuilt/package/06cb18a658a30c9ee232f93db1f894f286607974/include -isystem /home/curt/.conan/data/torch/1.6.0/curt/prebuilt/package/06cb18a658a30c9ee232f93db1f894f286607974/include/torch/csrc/api/include -isystem /usr/local/cuda/include -D_GLIBCXX_USE_CXX11_ABI=1 -std=c++14 -std=gnu++1z -o CMakeFiles/main.dir/main.cpp.o -c /data/projects/c-first/torch-simple/main.cpp

    # link
    /usr/bin/c++ CMakeFiles/main.dir/main.cpp.o -o main  
    -Wl,-rpath,
        /home/curt/.conan/data/torch/1.6.0/curt/prebuilt/package/06cb18a658a30c9ee232f93db1f894f286607974/lib:/usr/local/cuda/lib64/stubs:/usr/local/cuda/lib64 
        /usr/local/cuda/lib64/stubs/libcuda.so 
        /usr/local/cuda/lib64/libnvrtc.so
        /usr/local/cuda/lib64/libnvToolsExt.so 
        /usr/local/cuda/lib64/libcudart.so 
        /home/curt/.conan/data/torch/1.6.0/curt/prebuilt/package/06cb18a658a30c9ee232f93db1f894f286607974/lib/libc10.so
        /home/curt/.conan/data/torch/1.6.0/curt/prebuilt/package/06cb18a658a30c9ee232f93db1f894f286607974/lib/libc10_cuda.so
        /home/curt/.conan/data/torch/1.6.0/curt/prebuilt/package/06cb18a658a30c9ee232f93db1f894f286607974/lib/libtorch.so                   
    -Wl,--no-as-needed,
        /home/curt/.conan/data/torch/1.6.0/curt/prebuilt/package/06cb18a658a30c9ee232f93db1f894f286607974/lib/libtorch_cpu.so  
    -Wl,--no-as-needed,
        /home/curt/.conan/data/torch/1.6.0/curt/prebuilt/package/06cb18a658a30c9ee232f93db1f894f286607974/lib/libtorch_cuda.so 
    -Wl,--no-as-needed,
        /home/curt/.conan/data/torch/1.6.0/curt/prebuilt/package/06cb18a658a30c9ee232f93db1f894f286607974/lib/libtorch.so 
    -Wl,--as-needed 
        /home/curt/.conan/data/torch/1.6.0/curt/prebuilt/package/06cb18a658a30c9ee232f93db1f894f286607974/lib/libc10_cuda.so 
        /home/curt/.conan/data/torch/1.6.0/curt/prebuilt/package/06cb18a658a30c9ee232f93db1f894f286607974/lib/libc10.so 
        /usr/local/cuda/lib64/libcufft.so 
        /usr/local/cuda/lib64/libcurand.so 
        /usr/lib/x86_64-linux-gnu/libcublas.so 
        /usr/lib/x86_64-linux-gnu/libcudnn.so 
    -Wl,--as-needed 
        /usr/local/cuda/lib64/libnvToolsExt.so 
        /usr/local/cuda/lib64/libcudart.so     
    -Wl,--as-needed -lpthread
    
    
## ldd

    linux-vdso.so.1 (0x00007fff831ab000)
    /lib64/ld-linux-x86-64.so.2 (0x00007fc15a3b4000)
    libstdc++.so.6 => /usr/lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007fc109fa6000)
    libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007fc109d8e000)
    libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007fc10999d000)
    libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007fc1095ff000)
    librt.so.1 => /lib/x86_64-linux-gnu/librt.so.1 (0x00007fc108ab5000)
    libdl.so.2 => /lib/x86_64-linux-gnu/libdl.so.2 (0x00007fc1088b1000)
    libpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0 (0x00007fc1091bb000)    

    libnvToolsExt.so.1 => /usr/local/cuda/lib64/libnvToolsExt.so.1 (0x00007fc10a5f8000)
    libcudart.so.10.2 => /usr/local/cuda/lib64/libcudart.so.10.2 (0x00007fc10a37a000)
    
    libc10.so => /home/curt/.conan/data/torch/1.6.0/curt/prebuilt/package/06cb18a658a30c9ee232f93db1f894f286607974/lib/libc10.so (0x00007fc159f26000)
    libc10_cuda.so => /home/curt/.conan/data/torch/1.6.0/curt/prebuilt/package/06cb18a658a30c9ee232f93db1f894f286607974/lib/libc10_cuda.so (0x00007fc1083f8000)
    libcudart-80664282.so.10.2 => /home/curt/.conan/data/torch/1.6.0/curt/prebuilt/package/06cb18a658a30c9ee232f93db1f894f286607974/lib/libcudart-80664282.so.10.2 (0x00007fc108630000)
    libnvToolsExt-3965bdd0.so.1 => /home/curt/.conan/data/torch/1.6.0/curt/prebuilt/package/06cb18a658a30c9ee232f93db1f894f286607974/lib/libnvToolsExt-3965bdd0.so.1 (0x00007fc1081ee000)
    libtorch_cpu.so => /home/curt/.conan/data/torch/1.6.0/curt/prebuilt/package/06cb18a658a30c9ee232f93db1f894f286607974/lib/libtorch_cpu.so (0x00007fc149a00000)
    libtorch_cuda.so => /home/curt/.conan/data/torch/1.6.0/curt/prebuilt/package/06cb18a658a30c9ee232f93db1f894f286607974/lib/libtorch_cuda.so (0x00007fc10aa03000)
    libtorch.so => /home/curt/.conan/data/torch/1.6.0/curt/prebuilt/package/06cb18a658a30c9ee232f93db1f894f286607974/lib/libtorch.so (0x00007fc10a801000)
    libgomp-75eea7e8.so.1 => /home/curt/.conan/data/torch/1.6.0/curt/prebuilt/package/06cb18a658a30c9ee232f93db1f894f286607974/lib/libgomp-75eea7e8.so.1 (0x00007fc1093da000)
    libtensorpipe.so => /home/curt/.conan/data/torch/1.6.0/curt/prebuilt/package/06cb18a658a30c9ee232f93db1f894f286607974/lib/libtensorpipe.so (0x00007fc108cbd000)
    
## Reference
- https://nglee.github.io/2018/10/11/Study-Linker-Options.html
    