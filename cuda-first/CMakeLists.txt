cmake_minimum_required(VERSION 3.15)
project(cuda-first LANGUAGES CUDA CXX)

add_executable(cuda-version main.cc gpu.cu)

add_executable(do_cpu do_cpu.cpp)
add_executable(do_gpu do_gpu.cu)

add_executable(cublas_mm cublas_mm.cu)
target_link_libraries(cublas_mm cublas)

add_executable(matrix_multiplication_by_rand matrix_multiplication_by_rand.cu)
target_link_libraries(matrix_multiplication_by_rand cublas curand)
