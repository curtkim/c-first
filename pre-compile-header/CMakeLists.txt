cmake_minimum_required(VERSION 3.16)
project(pch-cmake)

add_executable(main main.cpp)
target_precompile_headers(main PRIVATE hello.h)


