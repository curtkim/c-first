cmake_minimum_required(VERSION 3.15)
project(coro2)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_executable(fibo fibo.cpp)
target_link_libraries(fibo ${CONAN_LIBS})
