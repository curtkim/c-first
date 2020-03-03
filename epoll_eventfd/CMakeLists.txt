cmake_minimum_required (VERSION 3.2 FATAL_ERROR)
project (epoll)

set (CMAKE_CXX_STANDARD_REQUIRED ON)

foreach(APP
        eventfd_worker)
    message("${APP}")
    add_executable(${APP} ${APP}.c)
    target_link_libraries( ${APP} pthread )
endforeach()
