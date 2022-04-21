cmake_minimum_required(VERSION 3.16)
project(bvh-first LANGUAGES CXX)

find_package(OpenGL REQUIRED)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()


foreach(APP
        01_basic
        )
  add_executable(${APP} ${APP}.cpp ) # memory_tracking.cpp
  target_link_libraries(${APP} ${CONAN_LIBS} ${OPENGL_LIBRARY} X11)
  target_compile_features(${APP} PUBLIC cxx_std_17)
  target_compile_definitions(${APP} PUBLIC STB_IMAGE_IMPLEMENTATION)
endforeach()

