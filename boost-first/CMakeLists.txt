cmake_minimum_required(VERSION 2.8.12)
project(boost-first)

add_definitions("-std=c++11")

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_executable(thread_pool thread_pool.cpp)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wformat=0")
target_link_libraries(thread_pool ${CONAN_LIBS})
