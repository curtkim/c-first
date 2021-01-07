cmake_minimum_required(VERSION 3.5.1)
project(ipc1)
add_executable(ipc1 ipc1.cpp)
target_link_libraries(ipc1 PUBLIC pthread)