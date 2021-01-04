cmake_minimum_required(VERSION 3.5.1)
project(lockfree)
add_executable(lockfree lockfree.cpp)
target_link_libraries(lockfree PUBLIC pthread)
