cmake_minimum_required(VERSION 2.8.12)
project(conan-gtest)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_definitions("-std=c++17")

set(TEST_SOURCES
        test-factorial.cpp
        test-vector.cpp
        )

add_executable(test-runner test-runner.cpp ${TEST_SOURCES})
