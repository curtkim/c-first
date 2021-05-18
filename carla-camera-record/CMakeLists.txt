cmake_minimum_required(VERSION 3.19)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(carla-camera-record LANGUAGES CXX CUDA )

set(VIDEO_SDK_VERSION 11.0.10)
#set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()


foreach(APP 01_cam1_thread1_cvcolor)
    add_executable(${APP} ${APP}.cpp
            NvEncoder/NvEncoder.cpp NvEncoder/NvEncoderCuda.cpp)
    target_compile_features(${APP} PUBLIC cxx_std_17)
    target_include_directories(${APP} PUBLIC .)
    target_include_directories(${APP} PRIVATE /usr/local/cuda/include)
    target_include_directories(${APP} PRIVATE /opt/nvidia/Video_Codec_SDK_${VIDEO_SDK_VERSION}/Interface)
    target_link_directories(${APP} PUBLIC /opt/nvidia/Video_Codec_SDK_${VIDEO_SDK_VERSION}/Lib/linux/stubs/x86_64)
    target_link_directories(${APP} PUBLIC /usr/local/cuda/lib64)
    target_precompile_headers(${APP} PRIVATE precompile.hpp)
    target_link_libraries(${APP} PUBLIC ${CONAN_LIBS} nvcuvid nvidia-encode cuda cudart nppicc)
endforeach()


