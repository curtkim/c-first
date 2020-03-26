cmake_minimum_required (VERSION 3.2 FATAL_ERROR)
project (exception)

set (CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(first first.cpp)
set_property(TARGET first PROPERTY CXX_STANDARD 17)
