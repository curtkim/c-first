cmake_minimum_required (VERSION 3.2 FATAL_ERROR)
project (modern-cpp)

set (CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(modern-cpp main.cpp TestObject.h)
set_property(TARGET modern-cpp PROPERTY CXX_STANDARD 17)

add_executable(modern-thread thread_main.cpp)
set_property(TARGET modern-thread PROPERTY CXX_STANDARD 17)

add_executable(cpp17 cpp17.cpp)
set_property(TARGET cpp17 PROPERTY CXX_STANDARD 17)

add_executable(move_forward move_forward.cpp)
set_property(TARGET move_forward PROPERTY CXX_STANDARD 17)