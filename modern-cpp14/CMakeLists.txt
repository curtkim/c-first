cmake_minimum_required (VERSION 3.2 FATAL_ERROR)
project (modern-cpp14)

set (CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(modern-cpp14 main.cpp )
set_property(TARGET modern-cpp14 PROPERTY CXX_STANDARD 14)
