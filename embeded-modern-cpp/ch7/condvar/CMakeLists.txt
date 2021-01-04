cmake_minimum_required(VERSION 3.5.1)
project(condvar)
add_executable(condvar condvar.cpp)
target_link_libraries(condvar PUBLIC pthread)