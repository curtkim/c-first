cmake_minimum_required(VERSION 2.8.12)
project(conan-imgui)

add_definitions("-std=c++17")
add_definitions(-DUSE_OPENCV)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_executable(classify classify.cpp)
target_link_libraries(classify ${CONAN_LIBS})
