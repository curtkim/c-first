cmake_minimum_required(VERSION 3.15)
project(coroutines)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_executable(main main.cpp)
target_compile_options(main PRIVATE -fcoroutines)

add_executable(echo_server echo_server.cpp)
target_compile_options(echo_server PRIVATE -fcoroutines)
target_compile_definitions(echo_server PUBLIC ASIO_HAS_CO_AWAIT)
target_compile_definitions(echo_server PUBLIC ASIO_HAS_STD_COROUTINE)
