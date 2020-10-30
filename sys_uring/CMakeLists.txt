cmake_minimum_required (VERSION 3.15 FATAL_ERROR)
project (sys_uring)

foreach(APP 01_cat_uring 04_capability uring-cp uring-test webserver )
    add_executable(${APP} ${APP}.c)
    target_link_libraries(${APP} uring )
    target_compile_definitions(${APP} PUBLIC _GNU_SOURCE)
endforeach()
