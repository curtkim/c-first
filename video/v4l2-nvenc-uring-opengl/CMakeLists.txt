cmake_minimum_required(VERSION 3.19)
project(v4l2-nvenc-uring-opengl LANGUAGES CXX C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(VIDEO_SDK_VERSION 11.0.10)
set(CUDA_DYNAMIC_LIBRARY cuda nvrtc nvToolsExt cudart cufft curand cudnn cublas)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_library(capture_utils utils/capture_utils.cpp)
add_library(opengl_utils utils/opengl_utils.cpp)

add_library(detr detr.cpp)
target_link_libraries(detr PUBLIC ${CONAN_LIBS})


# 2 nvidia env/dec
foreach(APP
        01_v4l2_opengl
        02_v4l2_nvenc
        03_v4l2_nvenc_opengl
        04_v4l2_nvenc_predict_opengl)
    add_executable(${APP} ${APP}.cpp
            NvEncoder/NvEncoder.cpp NvEncoder/NvEncoderCuda.cpp)
    target_include_directories(${APP} PRIVATE /usr/local/cuda/include)
    target_include_directories(${APP} PRIVATE /opt/nvidia/Video_Codec_SDK_${VIDEO_SDK_VERSION}/Interface)
    target_link_directories(${APP} PUBLIC /opt/nvidia/Video_Codec_SDK_${VIDEO_SDK_VERSION}/Lib/linux/stubs/x86_64)
    target_link_directories(${APP} PUBLIC /usr/local/cuda/lib64)
    target_link_libraries(${APP} PUBLIC ${CONAN_LIBS}
            ${CUDA_DYNAMIC_LIBRARY}
            nvcuvid nvidia-encode
            nppicc nppig)
    target_link_libraries(${APP} PUBLIC capture_utils opengl_utils)
    target_compile_features(${APP} PUBLIC cxx_std_17)
    set_target_properties(${APP} PROPERTIES LINK_WHAT_YOU_USE "TRUE")
endforeach()

target_link_libraries(04_v4l2_nvenc_predict_opengl PUBLIC detr)

add_executable(temp temp.cpp)