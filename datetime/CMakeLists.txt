cmake_minimum_required (VERSION 3.2 FATAL_ERROR)
project (datetime)

set (CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(main main.cpp)
set_property(TARGET main PROPERTY CXX_STANDARD 17)
#target_link_libraries( modern-thread pthread )
