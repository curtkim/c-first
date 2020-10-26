cmake_minimum_required(VERSION 3.15)
project(tensorrt-first LANGUAGES CXX)

#include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
#conan_basic_setup()
foreach(FILE mnist.prototxt mnist.caffemodel mnist_mean.binaryproto 7.pgm)
    file(COPY ${FILE} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)
endforeach()

add_executable(main main.cpp
        common/logger.cpp
)
target_include_directories(main PRIVATE common)
target_include_directories(main PRIVATE /usr/local/cuda/include)
target_link_directories(main PUBLIC /usr/local/cuda/lib64 /usr/local/TensorRT-7.1.3.4/lib)
target_link_libraries(main PUBLIC nvinfer nvparsers nvinfer_plugin nvonnxparser myelin nvrtc cudart cublas cudnn rt dl pthread)
