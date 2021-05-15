cmake_minimum_required(VERSION 3.15)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(video-nvidia-codec LANGUAGES CXX CUDA )

set(VIDEO_SDK_VERSION 11.0.10)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_executable(AppEncCuda AppEncCuda.cpp NvEncoder/NvEncoder.cpp NvEncoder/NvEncoderCuda.cpp NvEncoder/NvEncoderOutputInVidMemCuda.cpp Utils/crc.cu)
add_executable(AppEncGL AppEncGL.cpp NvEncoder/NvEncoder.cpp NvEncoder/NvEncoderGL.cpp GraphicsUtils.cpp)
add_executable(AppDec AppDec.cpp NvDecoder/NvDecoder.cpp)
add_executable(encode_bare encode_bare.cpp)
add_executable(simpleEncoder simpleEncoder.cpp NvEncoder/NvEncoder.cpp NvEncoder/NvEncoderCuda.cpp)
add_executable(simpleEncoder_test_video simpleEncoder_test_video.cpp NvEncoder/NvEncoder.cpp NvEncoder/NvEncoderCuda.cpp)
add_executable(rxcppEncoder rxcppEncoder.cpp NvEncoder/NvEncoder.cpp NvEncoder/NvEncoderCuda.cpp)


foreach(APP AppEncCuda AppEncGL AppDec encode_bare simpleEncoder simpleEncoder_test_video rxcppEncoder)
    target_compile_features(${APP} PUBLIC cxx_std_17)
    target_include_directories(${APP} PUBLIC .)
    target_include_directories(${APP} PRIVATE /usr/local/cuda/include)
    target_include_directories(${APP} PRIVATE /opt/nvidia/Video_Codec_SDK_${VIDEO_SDK_VERSION}/Interface)

    target_link_directories(${APP} PUBLIC /opt/nvidia/Video_Codec_SDK_${VIDEO_SDK_VERSION}/Lib/linux/stubs/x86_64)
    target_link_directories(${APP} PUBLIC /usr/local/cuda/lib64)
endforeach()

target_link_libraries(AppEncCuda PUBLIC ${CONAN_LIBS} nvcuvid nvidia-encode cuda cudart)
target_link_libraries(AppEncGL PUBLIC ${CONAN_LIBS} nvcuvid nvidia-encode cuda cudart X11)
target_link_libraries(AppDec PUBLIC ${CONAN_LIBS} nvcuvid nvidia-encode cuda cudart avformat avcodec avutil avfilter)
target_link_libraries(encode_bare PUBLIC ${CONAN_LIBS} nvcuvid nvidia-encode cuda cudart)
target_link_libraries(simpleEncoder PUBLIC ${CONAN_LIBS} nvcuvid nvidia-encode cuda cudart)
target_link_libraries(simpleEncoder_test_video PUBLIC ${CONAN_LIBS} nvcuvid nvidia-encode cuda cudart)
target_link_libraries(rxcppEncoder PUBLIC ${CONAN_LIBS} nvcuvid nvidia-encode cuda cudart)
