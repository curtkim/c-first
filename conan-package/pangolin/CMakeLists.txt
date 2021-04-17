cmake_minimum_required(VERSION 3.15)
project(cmake_wrapper)

set(CMAKE_VERBOSE_MAKEFILE ON)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_subdirectory("source_subfolder")
