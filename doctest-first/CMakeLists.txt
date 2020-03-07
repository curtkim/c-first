cmake_minimum_required(VERSION 2.8.12)
project(conan-gtest)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_definitions("-std=c++17")

add_executable(test-factorial test-factorial.cpp)
target_link_libraries(test-factorial)

add_executable(test-vector test-vector.cpp)
target_link_libraries(test-vector)
