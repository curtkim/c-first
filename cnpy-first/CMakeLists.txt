cmake_minimum_required(VERSION 3.15)
project(cnpy-first)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_definitions("-std=c++17")

add_executable(example example.cpp)
target_link_libraries(example PUBLIC ${CONAN_LIBS})
