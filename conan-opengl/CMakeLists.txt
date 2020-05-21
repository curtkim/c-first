cmake_minimum_required(VERSION 3.15)
project(conan-opengl)

find_package(OpenGL REQUIRED)
include_directories(.)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

foreach(FILE
    SimpleFragmentShader.fragmentshader
    SimpleVertexShader.vertexshader
    annulus.obj
    bunny.off
    lidar.ply)
  file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/${FILE} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)
endforeach()

foreach(APP main point point_multi mouse test-libigl )
  add_executable(${APP} ${APP}.cpp common/shader.cpp)
  target_link_libraries(${APP} ${CONAN_LIBS} ${OPENGL_LIBRARY} X11)
endforeach()
