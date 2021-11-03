cmake_minimum_required(VERSION 3.17)
project(unifex-first)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

#foreach(A ${CONAN_LIBS})
#    message(STATUS ${A})
#endforeach()

foreach(APP
        00_let
        01_when_all
        02_static_thread_pool
        03_schedule_with_subscheduler
        04_reduce_synchronous
        05_reduce_with_trampoline
        06_for_each_synchronous
        07_for_each_synchronous_simple
        08_transform_test

        11_via
        12_via_stream
        13_on_stream
        14_repeat_effect
        15_stop_immediately
        16_type_erased_stream
        17_stream_cancellation
        18_materialize

        21_io_epoll_basic
        22_io_epoll_test

        )
    add_executable(${APP} ${APP}.cpp)
    target_link_libraries(${APP} PUBLIC ${CONAN_LIBS} )
    target_compile_features(${APP} PUBLIC cxx_std_17)
endforeach()
