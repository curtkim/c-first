cmake_minimum_required(VERSION 3.15)
project(v4l2-first)

#include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
#conan_basic_setup()

add_executable(vidtest vidtest.c)

add_executable(my_capture_all my_capture_all.cpp)
add_executable(my_capture_mmap my_capture_mmap.cpp)
add_executable(my_capture_mmap_poll my_capture_mmap_poll.cpp)
add_executable(my_capture_mmap_epoll2 my_capture_mmap_epoll2.cpp)
add_executable(my_capture_mmap_epoll2_multi my_capture_mmap_epoll2_multi.cpp)

add_executable(my_capture_mmap_uring_poll my_capture_mmap_uring_poll.cpp)
target_link_libraries(my_capture_mmap_uring_poll PUBLIC uring)

add_library(capture OBJECT capture.cpp)