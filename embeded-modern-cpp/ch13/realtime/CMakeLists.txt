cmake_minimum_required(VERSION 3.5.1)
project(realtime)
add_executable(realtime realtime.cpp)
target_link_libraries(realtime PUBLIC pthread)

