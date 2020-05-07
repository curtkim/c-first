cmake_minimum_required(VERSION 3.15)
project(cmake_wrapper)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

find_package(Eigen3 REQUIRED)
add_subdirectory("source_subfolder")
