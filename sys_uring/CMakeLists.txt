cmake_minimum_required (VERSION 3.15 FATAL_ERROR)
project (sys_uring)

#include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
#conan_basic_setup()

foreach(APP 00_cat_raw )
    add_executable(${APP} ${APP}.c)
    target_link_libraries(${APP} pthread)
    target_compile_definitions(${APP} PUBLIC _GNU_SOURCE)
endforeach()

foreach(APP
        01_cat_uring 04_capability
        12_uring-cp-by-link_next_sqe 11_uring-test 13_fixed_buffers 14_link_req
        21_webserver
        31_eventfd 32_signalfd 33_timeout 34_rename 35_statx 36_file_update 37_poll-v-poll)
    add_executable(${APP} ${APP}.c)
    #target_link_libraries(${APP} ${CONAN_LIBS} pthread)
    target_link_libraries(${APP} uring pthread)
    target_compile_definitions(${APP} PUBLIC _GNU_SOURCE)
endforeach()

foreach(APP 51_inotify 52_eventfd)
    add_executable(${APP} ${APP}.cpp)
    target_link_libraries(${APP} uring pthread)
endforeach()
