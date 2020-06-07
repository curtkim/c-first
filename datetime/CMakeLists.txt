cmake_minimum_required (VERSION 3.2 FATAL_ERROR)
project (datetime)

set (CMAKE_CXX_STANDARD_REQUIRED ON)

foreach(APP main format)
    add_executable(${APP} ${APP}.cpp)
    set_property(TARGET ${APP} PROPERTY CXX_STANDARD 17)
    #target_link_libraries( modern-thread pthread )
endforeach()
