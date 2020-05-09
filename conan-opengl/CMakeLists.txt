cmake_minimum_required(VERSION 2.8.12)
project(conan-opengl)

add_definitions("-std=c++11")

find_package(OpenGL REQUIRED)
include_directories(.)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()


file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/SimpleFragmentShader.fragmentshader DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/SimpleVertexShader.vertexshader DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/annulus.obj DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/bunny.off DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)

foreach(APP main point mouse test-libigl)
  add_executable(${APP} ${APP}.cpp
      common/shader.cpp
      )
  target_link_libraries(${APP} ${CONAN_LIBS} ${OPENGL_LIBRARY} X11)
endforeach()
