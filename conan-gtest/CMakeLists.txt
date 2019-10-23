cmake_minimum_required(VERSION 2.8.12)
project(conan-gtest)

add_definitions("-std=c++11")

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

include_directories(mymath)
add_subdirectory(mymath)

add_executable(main main.cpp)
target_link_libraries(main mymath)
