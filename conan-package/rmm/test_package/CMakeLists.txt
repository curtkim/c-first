cmake_minimum_required(VERSION 3.18)
project(PackageTest CXX CUDA)

#set(CMAKE_VERBOSE_MAKEFILE ON)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

find_package(CUDAToolkit REQUIRED)

add_executable(example example.cpp)
target_include_directories(example PUBLIC /usr/local/cuda/include)
target_link_directories(example PUBLIC /usr/local/cuda/lib64)
target_link_libraries(example ${CONAN_LIBS})
target_compile_features(example PRIVATE cxx_std_17)
