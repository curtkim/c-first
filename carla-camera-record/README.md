
## HOWTO

    nsys profile -o _01_cam1_thread1_cvcolor.qdstrm --force-overwrite=true cmake-build-debug/bin/01_cam1_thread1_cvcolor
    nsys profile -o _02_cam1_thread1_npp.qdstrm --force-overwrite=true cmake-build-debug/bin/02_cam1_thread1_npp
    nsys profile -o _03_cam3_thread1_npp.qdstrm --force-overwrite=true cmake-build-debug/bin/03_cam3_thread1_npp

    nsys-ui 

## Reference
https://docs.nvidia.com/cuda/cuda-driver-api/driver-vs-runtime-api.html

### runtime api
- simple (implicit initialization, context management, and module management)
- Context management not exposed in runtime api
- Primary contexts are created as needed, one per device per process
- Within one process, all users of the runtime API will share the primary context, unless a context has been made current to each thread
- current context or primary context, can be synchronized with cudaDeviceSynchronize()
- destroyed with cudaDeviceReset().

### driver API
- more fine-grained control, especially over contexts and module loading
- CUDA clients can use the driver API to create and set the current context, and then use the runtime API to work with it.

## session개수 제한 풀기
https://github.com/keylase/nvidia-patch

## Reference
https://github.com/NVIDIA/NvPipe