cmake_minimum_required(VERSION 2.8.12)
project(eigen-first)

add_definitions("-std=c++11")

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

foreach(APP first transform quaternion roll_pitch_yaw)
    add_executable(${APP} ${APP}.cpp)
    target_link_libraries(${APP} ${CONAN_LIBS})
endforeach()
