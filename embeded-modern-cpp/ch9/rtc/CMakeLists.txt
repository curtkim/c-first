cmake_minimum_required(VERSION 3.5.1)
project(rtc)
add_executable(rtc rtc.cpp)
target_link_libraries(rtc PUBLIC pthread)
