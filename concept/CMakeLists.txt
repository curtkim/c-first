cmake_minimum_required (VERSION 3.15 FATAL_ERROR)
project (concept-cpp)

foreach(APP
        main range_in test-predicate norm
        )
    message("${APP}")
    add_executable(${APP} ${APP}.cpp)
    set_property(TARGET ${APP} PROPERTY CXX_STANDARD 20)
endforeach()
