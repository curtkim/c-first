cmake_minimum_required(VERSION 3.15)
project(openblas-first LANGUAGES CXX)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_executable(cblas_dgemm cblas_dgemm.cpp)
target_compile_features(cblas_dgemm PRIVATE cxx_std_17)
target_link_libraries(cblas_dgemm ${CONAN_LIBS})