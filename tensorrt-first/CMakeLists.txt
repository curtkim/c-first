cmake_minimum_required(VERSION 3.15)
project(tensorrt-first LANGUAGES CXX)

#set(TENSORRT_DIR /usr/local/TensorRT-8.2.3.0)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

set(TENSORRT_LIBS
        nvinfer
        nvparsers
        nvinfer_plugin
        nvonnxparser ) #myelin
set(CUDA_LIBS
        nvrtc
        cudart
        cublas
        cudnn)

foreach(FILE
        mnist.prototxt
        mnist.caffemodel
        mnist_mean.binaryproto
        7.pgm)
    file(COPY mnist_caffe/${FILE} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)
endforeach()

add_executable(infer_mnist mnist_caffe/infer_mnist.cpp
    common/logger.cpp
    mnist_caffe/sample_mnist.cpp
)
target_include_directories(infer_mnist PRIVATE common)
target_include_directories(infer_mnist PRIVATE /usr/local/cuda/include)
#target_include_directories(infer_mnist PRIVATE ${TENSORRT_DIR}/include)
target_link_directories(infer_mnist PUBLIC /usr/local/cuda/lib64) #${TENSORRT_DIR}/lib)
target_link_libraries(infer_mnist PUBLIC ${CONAN_LIBS} ${TENSORRT_LIBS} ${CUDA_LIBS} rt dl pthread)


add_executable(infer_resnet resnet/infer_resnet.cpp)
target_include_directories(infer_resnet PRIVATE common)
target_include_directories(infer_resnet PRIVATE /usr/local/cuda/include)
#target_include_directories(infer_resnet PRIVATE ${TENSORRT_DIR}/include)
target_link_directories(infer_resnet PUBLIC /usr/local/cuda/lib64 ) #${TENSORRT_DIR}/lib
target_link_libraries(infer_resnet PUBLIC ${CONAN_LIBS} ${TENSORRT_LIBS} ${CUDA_LIBS} rt dl pthread
    nppc nppig nppisu nppial nppidei nppim nppitc nppicc nppif nppist npps)

foreach(APP infer_add1 infer_add1_stream)
    add_executable(${APP} add1/${APP}.cpp)
    target_include_directories(${APP} PRIVATE /usr/local/cuda/include)
    target_link_directories(${APP} PUBLIC /usr/local/cuda/lib64) #${TENSORRT_DIR}/lib)
    target_link_libraries(${APP} PUBLIC ${CONAN_LIBS} ${TENSORRT_LIBS} ${CUDA_LIBS} rt dl pthread)
endforeach()
