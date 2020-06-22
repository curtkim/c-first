cmake_minimum_required(VERSION 3.15)
project(doctest-first)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_definitions("-std=c++17")

set(TEST_SOURCES
        test-factorial.cpp
        test-vector.cpp
        test-bitwise.cpp
        )

add_executable(test-runner test-runner.cpp ${TEST_SOURCES})
#add_doc_test(all-ests test-runner)