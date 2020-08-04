cmake_minimum_required (VERSION 3.15 FATAL_ERROR)
project (compiler_abi)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_executable(main main.cpp mystring.cpp)
