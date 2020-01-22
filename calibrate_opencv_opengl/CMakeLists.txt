cmake_minimum_required(VERSION 2.8.12)
project(calibrate-opencv-opengl)

add_definitions("-std=c++11")

find_package(OpenGL REQUIRED)
include_directories(.)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_executable(calibrate-opencv-opengl
    main.cpp
  )
target_link_libraries(calibrate-opencv-opengl ${CONAN_LIBS} ${OPENGL_LIBRARY})
