cmake_minimum_required (VERSION 3.2 FATAL_ERROR)
project (rxcpp-first)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

#get_cmake_property(_variableNames VARIABLES)
#list (SORT _variableNames)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()

foreach(APP 01_create 02_op 03_read_file 04_last_five
        05_create_thread 06_create_thread_interval
        09_complex 08_complex_step
        11_parallel 12_rxcurl_ex 13_rxrest
        20_coordination_list 21_observe_on 22_observe_on_runloop 25_synchronize_in 28_worker_schedule
        31_multi_subscribe 32_filter_with_latest
        51_value_move 52_dynamic_array
        61_create_asio_timer
        62_asio_stdin
        66_asio_timeout
        91_omega )
    add_executable(${APP} ${APP}.cpp)
    target_compile_features(${APP} PUBLIC cxx_std_17)
    target_link_libraries (${APP} ${CONAN_LIBS} pthread)
endforeach()

# example
foreach(APP merge merge_without_publish publish op_timer op_timeout ref_count defer)
    add_executable(${APP} example/${APP}.cpp)
    target_compile_features(${APP} PUBLIC cxx_std_17)
    target_link_libraries (${APP} ${CONAN_LIBS} pthread)
endforeach()
