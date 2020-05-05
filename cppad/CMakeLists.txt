cmake_minimum_required (VERSION 3.2 FATAL_ERROR)
project (cppad)

foreach(APP get_start)
    add_executable(${APP} ${APP}.cpp)
    set_property(TARGET ${APP} PROPERTY CXX_STANDARD 17)
    target_link_libraries( ${APP} pthread )
endforeach()
