cmake_minimum_required(VERSION 2.8.12)
project(conan-imgui)

add_definitions("-std=c++11")

find_package(OpenGL REQUIRED)
include_directories(.)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_executable(conan-imgui
    main.cpp
	imgui_impl_glfw.cpp
	imgui_impl_glfw.h
	imgui_impl_opengl2.cpp
	imgui_impl_opengl2.h
  )
target_link_libraries(conan-imgui ${CONAN_LIBS} ${OPENGL_LIBRARY})
