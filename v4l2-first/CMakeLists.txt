cmake_minimum_required(VERSION 3.15)
project(v4l2-first)

#include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
#conan_basic_setup()

add_executable(vidtest vidtest.c)
add_executable(my_capture_all my_capture_all.cpp)
add_executable(my_capture_mmap my_capture_mmap.cpp)
add_executable(my_capture_mmap_epoll my_capture_mmap_epoll.cpp)
add_executable(my_capture_mmap_epoll2 my_capture_mmap_epoll2.cpp)
add_executable(my_capture_mmap_epoll2_multi my_capture_mmap_epoll2_multi.cpp)
#add_executable(v4l2grab v4l2grab.c yuv.c)

add_library(capture OBJECT capture.cpp)