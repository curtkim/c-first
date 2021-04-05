project(PackageTest CXX)
cmake_minimum_required(VERSION 3.16)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup(TARGETS)

add_executable(test_package hs071_main.cpp hs071_nlp.cpp)
target_link_libraries(test_package PRIVATE ${CONAN_LIBS})

enable_testing()
add_test(NAME test_package
         WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
         COMMAND test_package)
