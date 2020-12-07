cmake_minimum_required (VERSION 3.15 FATAL_ERROR)
project (sys_uring)

foreach(APP 00_cat_raw 01_cat_uring 04_capability 12_uring-cp 11_uring-test
        21_webserver
        31_eventfd 32_signalfd 33_timeout 34_rename 35_statx 36_file_update)
    add_executable(${APP} ${APP}.c)
    target_link_libraries(${APP} uring pthread)
    target_compile_definitions(${APP} PUBLIC _GNU_SOURCE)
endforeach()
