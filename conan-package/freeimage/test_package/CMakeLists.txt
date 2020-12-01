project(PackageTest CXX)
cmake_minimum_required(VERSION 2.8.12)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup(TARGETS)

add_executable(test_package test_package.cpp)
target_link_libraries(test_package PRIVATE CONAN_PKG::freeimage)

enable_testing()
add_test(NAME test_package
         WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
         COMMAND test_package)
