cmake_minimum_required (VERSION 3.15 FATAL_ERROR)
project (sys_uring)

#include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
#conan_basic_setup()

foreach(APP 00_cat_raw 01_cat_uring 04_capability
        12_uring-cp 11_uring-test 13_fixed_buffers 14_link_req
        21_webserver
        31_eventfd 32_signalfd 33_timeout 34_rename 35_statx 36_file_update 37_poll-v-poll)
    add_executable(${APP} ${APP}.c)
    #target_link_libraries(${APP} ${CONAN_LIBS} pthread)
    target_link_libraries(${APP} uring pthread)
    target_compile_definitions(${APP} PUBLIC _GNU_SOURCE)
endforeach()

add_executable(51_inotify 51_inotify.cpp)
target_link_libraries(51_inotify uring pthread)

