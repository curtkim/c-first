cmake_minimum_required(VERSION 3.19)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(eigen-cuda CXX CUDA)


include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()


foreach(APP 01_first 02_second)
    add_executable(${APP} ${APP}.cu)
    target_include_directories(${APP} PRIVATE /usr/local/cuda/include)
    target_link_directories(${APP} PRIVATE /usr/local/cuda/lib64)
    target_link_libraries(${APP} PRIVATE cuda cudart)
    target_compile_features(${APP} PUBLIC cxx_std_17)
    target_compile_definitions(${APP} PRIVATE EIGEN_USE_GPU)

    target_compile_options(${APP} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
            --expt-relaxed-constexpr
            --expt-extended-lambda
            --default-stream per-thread
            --use_fast_math
            -Xcudafe "--diag_suppress=20011 --diag_suppress=20014"
            >) #-Xcudafe --display_error_number
endforeach()
