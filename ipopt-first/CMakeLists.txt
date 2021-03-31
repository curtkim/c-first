cmake_minimum_required(VERSION 3.16)
project(ipopt-first)

#include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
#conan_basic_setup()

add_executable(cpp-example cpp-example.cpp MyNLP.cpp)
target_link_libraries(cpp-example ipopt)

add_executable(hs071_main hs071_main.cpp hs071_nlp.cpp)
target_link_libraries(hs071_main ipopt)
