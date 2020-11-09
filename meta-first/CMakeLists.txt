cmake_minimum_required (VERSION 3.15 FATAL_ERROR)
project (meta-first )

foreach(APP 01_add 02_relaxNG 03_decay 04_enable_if 05_is_lvalue_reference
        11_generic_copy 12_tag_dispatching)
    add_executable(${APP} ${APP}.cpp)
    #target_link_libraries(${APP} uring )
    #target_compile_definitions(${APP} PUBLIC _GNU_SOURCE)
    target_compile_features(${APP} PUBLIC cxx_std_17)
endforeach()
