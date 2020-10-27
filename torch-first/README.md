## reference
- https://github.com/tobiascz/MNIST_Pytorch_python_and_capi
- https://github.com/IlyaOvodov/TorchScriptTutorial

## TODO
- 현재 cpu를 사용함. GPU를 사용하도록 변경

## conan
https://github.com/conan-io/wishlist/issues/187
https://github.com/forwardmeasure/conan/tree/master/recipes/conan-libtorch

## INSTALLING C++ DISTRIBUTIONS OF PYTORCH
https://pytorch.org/cppdocs/installing.html

    -DCMAKE_PREFIX_PATH=/home/curt/projects/c/libtorch
    

## Link

/usr/bin/c++     -g        CMakeFiles/04_cuda_detect.dir/04_cuda_detect.cpp.o  -o bin/04_cuda_detect   
    -L/home/curt/.conan/data/torch/1.6.0/curt/prebuilt/package/06cb18a658a30c9ee232f93db1f894f286607974/lib  
    -L/usr/local/cuda/lib64  
    -Wl,-rpath,/home/curt/.conan/data/torch/1.6.0/curt/prebuilt/package/06cb18a658a30c9ee232f93db1f894f286607974/lib:/usr/local/cuda/lib64 
    -lcudart 
    -lnvToolsExt 
    -lCaffe2_perfkernels_avx 
    -lCaffe2_perfkernels_avx2 
    -lCaffe2_perfkernels_avx512 
    -lXNNPACK 
    -lasmjit 
    -lbenchmark 
    -lbenchmark_main 
    -lc10 
    -lc10_cuda 
    -lc10d 
    -lc10d_cuda_test 
    -lcaffe2_detectron_ops_gpu 
    -lcaffe2_module_test_dynamic 
    -lcaffe2_nvrtc 
    -lcaffe2_observers 
    -lcaffe2_protos 
    -lclog 
    -lcpuinfo 
    -lcpuinfo_internals 
    -ldnnl 
    -lfbgemm 
    -lfmt 
    -lfoxi_loader 
    -lgloo 
    -lgloo_cuda 
    -lgmock 
    -lgmock_main 
    -lgtest 
    -lgtest_main 
    -lmkldnn 
    -lnnpack 
    -lnnpack_reference_layers 
    -lnvrtc-builtins 
    -lonnx 
    -lonnx_proto 
    -lprotobuf 
    -lprotobuf-lite 
    -lprotoc 
    -lpthreadpool 
    -lpytorch_qnnpack 
    -lqnnpack 
    -lshm 
    -ltensorpipe 
    -ltorch 
    -ltorch_cpu 
    -ltorch_cuda 
    -ltorch_global_deps 
    -ltorch_python 
    -luv 
    -luv_a 
    -lc10 
    -ltensorpipe 
    -ltorch_cpu 
    -lgomp 
    -ltorch_cuda 
    -ltorch 
    -lc10_cuda 
    -lc10 
    -lc10d 
    -lc10d_cuda_test 
    -lcaffe2_detectron_ops_gpu 
    -lcaffe2_module_test_dynamic 
    -lcaffe2_nvrtc -lcaffe2_observers 
    -lcaffe2_protos 
    -lclog 
    -lcpuinfo 
    -lcpuinfo_internals 
    -ldnnl 
    -lfbgemm 
    -lfmt 
    -lfoxi_loader 
    -lgloo 
    -lgloo_cuda 
    -lgmock 
    -lgmock_main 
    -lgtest 
    -lgtest_main 
    -lmkldnn 
    -lnnpack 
    -lnnpack_reference_layers 
    -lnvrtc-builtins 
    -lonnx 
    -lonnx_proto 
    -lprotobuf 
    -lprotobuf-lite 
    -lprotoc 
    -lpthreadpool 
    -lpytorch_qnnpack 
    -lqnnpack 
    -lshm 
    -ltensorpipe 
    -ltorch_cpu 
    -ltorch_global_deps 
    -ltorch_python 
    -luv 
    -luv_a 
    -lgomp    
   