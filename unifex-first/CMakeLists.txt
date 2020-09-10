cmake_minimum_required(VERSION 3.17)
project(unifex-first)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

#foreach(A ${CONAN_LIBS})
#    message(STATUS ${A})
#endforeach()

foreach(APP when_all static_thread_pool reduce_with_trampoline schedule_with_subscheduler let for_each_synchronous io_epoll_test
        transform_test)
    add_executable(${APP} ${APP}.cpp)
    target_link_libraries(${APP} PUBLIC ${CONAN_LIBS} )
    target_compile_features(${APP} PUBLIC cxx_std_20)
endforeach()