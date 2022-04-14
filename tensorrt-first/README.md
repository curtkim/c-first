## build
- header file(ex NvInfer.h) is located at /usr/include/x86_64-linux-gnu 
- lib file(ex libnvinfer.so) is located at /lib/x86_64-linux-gnu/libnvinfer.so
- target_link_directories에 tensorrt_dir/lib 포함시킬 필요없다.

## main
    caffemodel을 로딩해서 TensorRT를 실행한다. 

    cmake-build-debug/bin/main

    [04/12/2022-13:58:25] [I] Building and running a GPU inference engine for MNIST
    [04/12/2022-13:58:25] [I] [TRT] [MemUsageChange] Init CUDA: CPU +458, GPU +0, now: CPU 469, GPU 1279 (MiB)
    [04/12/2022-13:58:26] [I] [TRT] [MemUsageSnapshot] Begin constructing builder kernel library: CPU 469 MiB, GPU 1279 MiB
    [04/12/2022-13:58:26] [I] [TRT] [MemUsageSnapshot] End constructing builder kernel library: CPU 623 MiB, GPU 1323 MiB
    [04/12/2022-13:58:26] [I] [TRT] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +809, GPU +350, now: CPU 1434, GPU 1673 (MiB)
    [04/12/2022-13:58:27] [I] [TRT] [MemUsageChange] Init cuDNN: CPU +659, GPU +266, now: CPU 2093, GPU 1939 (MiB)
    [04/12/2022-13:58:27] [I] [TRT] Local timing cache in use. Profiling results in this builder pass will not be stored.
    [04/12/2022-13:58:39] [I] [TRT] Some tactics do not have sufficient workspace memory to run. Increasing workspace size may increase performance, please check verbose output.
    [04/12/2022-13:58:39] [I] [TRT] Detected 1 inputs and 1 output network tensors.
    [04/12/2022-13:58:39] [I] [TRT] Total Host Persistent Memory: 8448
    [04/12/2022-13:58:39] [I] [TRT] Total Device Persistent Memory: 1626624
    [04/12/2022-13:58:39] [I] [TRT] Total Scratch Memory: 0
    [04/12/2022-13:58:39] [I] [TRT] [MemUsageStats] Peak memory usage of TRT CPU/GPU memory allocators: CPU 1 MiB, GPU 13 MiB
    [04/12/2022-13:58:39] [I] [TRT] [BlockAssignment] Algorithm ShiftNTopDown took 0.036539ms to assign 3 blocks to 8 nodes requiring 57857 bytes.
    [04/12/2022-13:58:39] [I] [TRT] Total Activation Memory: 57857
    [04/12/2022-13:58:39] [I] [TRT] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +0, GPU +8, now: CPU 2996, GPU 2351 (MiB)
    [04/12/2022-13:58:39] [I] [TRT] [MemUsageChange] Init cuDNN: CPU +0, GPU +10, now: CPU 2996, GPU 2361 (MiB)
    [04/12/2022-13:58:39] [I] [TRT] [MemUsageChange] TensorRT-managed allocation in building engine: CPU +0, GPU +4, now: CPU 0, GPU 4 (MiB)
    [04/12/2022-13:58:39] [I] [TRT] [MemUsageChange] Init cuBLAS/cuBLASLt: CPU +0, GPU +8, now: CPU 2841, GPU 2317 (MiB)
    [04/12/2022-13:58:39] [I] [TRT] [MemUsageChange] Init cuDNN: CPU +0, GPU +8, now: CPU 2841, GPU 2325 (MiB)
    [04/12/2022-13:58:39] [I] [TRT] [MemUsageChange] TensorRT-managed allocation in IExecutionContext creation: CPU +0, GPU +1, now: CPU 0, GPU 5 (MiB)
    [04/12/2022-13:58:39] [I] Input:
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@++-  .@@@@@@@
    @@@@@@@@@@@@#+-      .@@@@@@
    @@@@@@@@%:..   :     .@@@@@@
    @@@@@@@#.     %@%#.  :@@@@@@
    @@@@@@@:    +#@@@=  .@@@@@@@
    @@@@@@#   .#@@@@@:  =@@@@@@@
    @@@@@@:  :#@@@@@*  :#@@@@@@@
    @@@@@@*.:%@@@@@+  .%@@@@@@@@
    @@@@@@@@@@@@@@*   +@@@@@@@@@
    @@@@@@@@@@@@@@   +@@@@@@@@@@
    @@@@@@@@@@@@@=   %@@@@@@@@@@
    @@@@@@@@@@@@@:  +@@@@@@@@@@@
    @@@@@@@@@@@@+  -@@@@@@@@@@@@
    @@@@@@@@@@@-  +@@@@@@@@@@@@@
    @@@@@@@@@@%  .@@@@@@@@@@@@@@
    @@@@@@@@@%- .%@@@@@@@@@@@@@@
    @@@@@@@@@-  +@@@@@@@@@@@@@@@
    @@@@@@@@%: =%@@@@@@@@@@@@@@@
    @@@@@@@@= .%@@@@@@@@@@@@@@@@
    @@@@@@@@# *@@@@@@@@@@@@@@@@@
    @@@@@@@@@@@@@@@@@@@@@@@@@@@@

    [04/12/2022-13:58:39] [I] Output:
    0:
    1:
    2:
    3:
    4:
    5:
    6:
    7: **********
    8:
    9:

    &&&& PASSED TensorRT.sample_mnist # cmake-build-debug/bin/main

    Process finished with exit code 0


## infer_resnet

    python pytorch_model.py
    cmake-build-debug/bin/infer_resnet resnet/resnet50.onnx resnet/turkish_coffee.jpg
    
    ----------------------------------------------------------------
    Input filename:   resnet50.onnx
    ONNX IR version:  0.0.6
    Opset version:    9
    Producer name:    pytorch
    Producer version: 1.7
    Domain:           
    Model version:    0
    Doc string:       
    ----------------------------------------------------------------
    class: web site, website, internet site, site | confidence: 5.84855% | index: 916
    class: cleaver, meat cleaver, chopper | confidence: 3.78223% | index: 499
    ...