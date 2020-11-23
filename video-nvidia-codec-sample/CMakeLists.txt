cmake_minimum_required(VERSION 3.15)
project(video-nvidia-codec LANGUAGES CXX CUDA )

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()



# 2 nvidia env/dec
add_executable(encode AppEncCuda.cpp NvEncoder/NvEncoder.cpp NvEncoder/NvEncoderCuda.cpp NvEncoder/NvEncoderOutputInVidMemCuda.cpp Utils/crc.cu)
target_include_directories(encode PUBLIC .)
target_link_directories(encode PUBLIC /home/curt/projects/nvidia/Video_Codec_SDK_10.0.26/Lib/linux/stubs/x86_64)
target_link_directories(encode PUBLIC /usr/local/cuda/lib64)
target_link_libraries(encode PUBLIC ${CONAN_LIBS} nvcuvid nvidia-encode cuda cudart)
target_compile_features(encode PUBLIC cxx_std_17)
target_include_directories(encode PRIVATE /usr/local/cuda/include)
target_include_directories(encode PRIVATE /home/curt/projects/nvidia/Video_Codec_SDK_10.0.26/Interface)


add_executable(decode AppDec.cpp NvDecoder/NvDecoder.cpp)
target_include_directories(decode PUBLIC .)
target_link_directories(decode PUBLIC /home/curt/projects/nvidia/Video_Codec_SDK_10.0.26/Lib/linux/stubs/x86_64)
target_link_directories(decode PUBLIC /usr/local/cuda/lib64)
target_link_libraries(decode PUBLIC ${CONAN_LIBS} nvcuvid nvidia-encode cuda cudart avformat avcodec avutil avfilter)
target_compile_features(decode PUBLIC cxx_std_17)
target_include_directories(decode PRIVATE /usr/local/cuda/include)
target_include_directories(decode PRIVATE /home/curt/projects/nvidia/Video_Codec_SDK_10.0.26/Interface)
