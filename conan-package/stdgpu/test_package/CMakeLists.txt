cmake_minimum_required(VERSION 3.15)
project(PackageTest LANGUAGES CXX CUDA)

#set(CMAKE_VERBOSE_MAKEFILE ON)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

find_package(Threads REQUIRED)

add_executable(example example.cu)
target_link_libraries(example ${CONAN_LIBS})
target_compile_features(example PRIVATE cxx_std_17)
#target_include_directories(example PUBLIC /usr/local/cuda/include/)
#target_link_directories(example PUBLIC /usr/local/cuda/lib64)
#target_link_libraries(example cuda cudart)