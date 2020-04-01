cmake_minimum_required(VERSION 2.8.12)
project(range)

add_definitions("-std=c++17")

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_executable(views views.cpp)
target_link_libraries(views) # header only library

add_executable(algorithm algorithm.cpp)
target_link_libraries(algorithm)
