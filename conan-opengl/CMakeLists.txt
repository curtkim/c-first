cmake_minimum_required(VERSION 2.8.12)
project(conan-opengl)

add_definitions("-std=c++11")

find_package(OpenGL REQUIRED)
include_directories(.)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_executable(conan-opengl
    main.cpp
    common/shader.cpp
    common/shader.hpp
    SimpleFragmentShader.fragmentshader
    SimpleVertexShader.vertexshader
  )
target_link_libraries(conan-opengl ${CONAN_LIBS} ${OPENGL_LIBRARY})
