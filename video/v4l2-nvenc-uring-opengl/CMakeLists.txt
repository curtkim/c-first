cmake_minimum_required(VERSION 3.19)
project(v4l2-nvenc-uring-opengl LANGUAGES CXX C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(VIDEO_SDK_VERSION 11.0.10)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_library(capture_utils utils/capture_utils.cpp)
add_library(opengl_utils utils/opengl_utils.cpp)


# 2 nvidia env/dec
foreach(APP
        01_v4l2_opengl
        02_v4l2_nvenc_opengl)
    add_executable(${APP} ${APP}.cpp
            NvEncoder/NvEncoder.cpp NvEncoder/NvEncoderCuda.cpp)
    target_include_directories(${APP} PRIVATE /usr/local/cuda/include)
    target_include_directories(${APP} PRIVATE /opt/nvidia/Video_Codec_SDK_${VIDEO_SDK_VERSION}/Interface)
    target_link_directories(${APP} PUBLIC /opt/nvidia/Video_Codec_SDK_${VIDEO_SDK_VERSION}/Lib/linux/stubs/x86_64)
    target_link_directories(${APP} PUBLIC /usr/local/cuda/lib64)
    target_link_libraries(${APP} PUBLIC ${CONAN_LIBS}
            nvcuvid nvidia-encode cuda cudart
            nppicc
            nvToolsExt)
    target_link_libraries(${APP} PUBLIC capture_utils opengl_utils)
    target_compile_features(${APP} PUBLIC cxx_std_17)
endforeach()


