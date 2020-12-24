cmake_minimum_required(VERSION 3.15)
project(video-nvidia-codec LANGUAGES CXX C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

# 2 nvidia env/dec
foreach(APP encode encode_two_lamdba)
    add_executable(${APP} ${APP}.cpp)
    target_link_directories(${APP} PUBLIC /home/curt/projects/nvidia/Video_Codec_SDK_10.0.26/Lib/linux/stubs/x86_64)
    target_link_directories(${APP} PUBLIC /usr/local/cuda/lib64)
    target_link_libraries(${APP} PUBLIC ${CONAN_LIBS} nvcuvid nvidia-encode cuda cudart)
    target_compile_features(${APP} PUBLIC cxx_std_17)
    target_include_directories(${APP} PRIVATE /usr/local/cuda/include)
    target_include_directories(${APP} PRIVATE /home/curt/projects/nvidia/Video_Codec_SDK_10.0.26/Interface)
endforeach()

add_executable(yuv2rgb yuv2rgb)
target_link_libraries(yuv2rgb PUBLIC ${CONAN_LIBS})
target_compile_features(yuv2rgb PUBLIC cxx_std_17)
