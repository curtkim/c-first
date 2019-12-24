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
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/SimpleFragmentShader.fragmentshader DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/SimpleVertexShader.vertexshader DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)

add_executable(point point.cpp)
target_link_libraries(point ${CONAN_LIBS} ${OPENGL_LIBRARY})

add_executable(mouse mouse.cpp)
target_link_libraries(mouse ${CONAN_LIBS} ${OPENGL_LIBRARY})
