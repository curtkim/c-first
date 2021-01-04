cmake_minimum_required(VERSION 3.5.1)
project(memorder)
add_executable(memorder memorder.cpp)
target_link_libraries(memorder PUBLIC pthread)