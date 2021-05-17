cmake_minimum_required(VERSION 3.19)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(video-yuv LANGUAGES CUDA C CXX)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

foreach(APP npp_image_rgb2yuv444)
    add_executable(${APP} ${APP}.cu)
    target_include_directories(${APP} PRIVATE /usr/local/cuda/include)
    target_link_directories(${APP} PRIVATE /usr/local/cuda/lib64)
    target_link_libraries(${APP} PUBLIC ${CONAN_LIBS})
    target_link_libraries(${APP} PRIVATE cuda cudart nppicc)
endforeach()

set_target_properties(
        npp_image_rgb2yuv444
        PROPERTIES CUDA_ARCHITECTURES 61)