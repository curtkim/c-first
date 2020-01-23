cmake_minimum_required(VERSION 2.8.12)
project(eigen-first)

add_definitions("-std=c++11")

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_executable(first first.cpp)
target_link_libraries(first ${CONAN_LIBS})
