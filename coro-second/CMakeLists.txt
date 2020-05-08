cmake_minimum_required(VERSION 3.15)
project(coroutines)

add_executable(main main.cpp)
target_compile_options(main PRIVATE -fcoroutines)
