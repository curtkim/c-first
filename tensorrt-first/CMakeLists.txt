cmake_minimum_required(VERSION 3.15)
project(tensorrt-first LANGUAGES CXX)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

set(TENSORRT_LIBS nvinfer nvparsers nvinfer_plugin nvonnxparser myelin)
set(CUDA_LIBS nvrtc cudart cublas cudnn)

foreach(FILE mnist.prototxt mnist.caffemodel mnist_mean.binaryproto 7.pgm)
    file(COPY ${FILE} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)
endforeach()

add_executable(main main.cpp
    common/logger.cpp
)
add_executable(infer_resnet infer_resnet.cpp)

foreach(APP main infer_resnet)
    target_include_directories(${APP} PRIVATE common)
    target_include_directories(${APP} PRIVATE /usr/local/cuda/include)
    target_include_directories(${APP} PRIVATE /usr/local/TensorRT-7.1.3.4/include)
    target_link_directories(${APP} PUBLIC /usr/local/cuda/lib64 /usr/local/TensorRT-7.1.3.4/lib)
    target_link_libraries(${APP} PUBLIC ${CONAN_LIBS} ${TENSORRT_LIBS} ${CUDA_LIBS}  rt dl pthread)
        #nppc nppicom nppig nppisu nppial nppidei nppim nppitc nppicc nppif nppist npps
endforeach()


