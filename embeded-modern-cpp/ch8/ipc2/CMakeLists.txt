cmake_minimum_required(VERSION 3.5.1)
project(ipc2)
add_executable(ipc2 ipc2.cpp)
target_link_libraries(ipc2 PUBLIC pthread)