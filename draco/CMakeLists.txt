cmake_minimum_required(VERSION 3.15)
project(draco-first)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_definitions("-std=c++17")

add_executable(encode encode.cpp)
target_link_libraries(encode ${CONAN_LIBS})