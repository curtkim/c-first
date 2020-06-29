cmake_minimum_required(VERSION 3.15)
project(shapefile-read)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_definitions("-std=c++17")

add_executable(main main.cpp)
#target_link_libraries(main)

add_executable(main2 main2.cpp)
target_link_libraries(main2 ${CONAN_LIBS})