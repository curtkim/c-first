cmake_minimum_required(VERSION 2.8.12)
project(eigen-first)

add_definitions("-std=c++11")

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

foreach(APP 01_first 02_transform 03_quaternion 04_roll_pitch_yaw 05_c_data
        12_raw_buffer 13_as_parameter)
    add_executable(${APP} ${APP}.cpp)
    target_link_libraries(${APP} ${CONAN_LIBS})
endforeach()

add_library(mallochook SHARED mallochook.c)


# with malloc hook
add_executable(11_memory 11_memory.cpp)
target_link_libraries(11_memory ${CONAN_LIBS} mallochook dl)