cmake_minimum_required(VERSION 3.17)
project(unifex-first)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

#foreach(A ${CONAN_LIBS})
#    message(STATUS ${A})
#endforeach()

foreach(APP when_all static_thread_pool schedule_with_subscheduler let
        reduce_synchronous reduce_with_trampoline
        for_each_synchronous for_each_synchronous_simple
        io_epoll_test
        transform_test
        via
        produce_on_consume_via
        for_each_via_thread_scheduler)
    add_executable(${APP} ${APP}.cpp)
    target_link_libraries(${APP} PUBLIC ${CONAN_LIBS} )
    target_compile_features(${APP} PUBLIC cxx_std_20)
endforeach()