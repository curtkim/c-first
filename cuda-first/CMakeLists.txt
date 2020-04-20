cmake_minimum_required(VERSION 3.15)
project(cuda-first LANGUAGES CUDA CXX)

add_executable(cuda-version main.cc gpu.cu)

add_executable(do_cpu do_cpu.cpp)
add_executable(do_gpu do_gpu.cu)
