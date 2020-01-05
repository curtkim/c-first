cmake_minimum_required (VERSION 3.2 FATAL_ERROR)
project (modern-cpp)

set (CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(producer producer.cpp)
add_executable(consumer consumer.cpp)

add_executable(graceful graceful.cpp)
target_link_libraries( graceful pthread )