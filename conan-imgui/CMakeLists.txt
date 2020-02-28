cmake_minimum_required(VERSION 2.8.12)
project(conan-imgui)

add_definitions("-std=c++11")
# opengl loader를 GLAD로 설정한다.
add_definitions(-DIMGUI_IMPL_OPENGL_LOADER_GLAD)

find_package(OpenGL REQUIRED)
include_directories(.)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_executable(main_glfw_opengl2
		main_glfw_opengl2.cpp
		bindings/imgui_impl_glfw.cpp
		bindings/imgui_impl_glfw.h
		bindings/imgui_impl_opengl2.cpp
		bindings/imgui_impl_opengl2.h
  )
target_link_libraries(main_glfw_opengl2 ${CONAN_LIBS} ${OPENGL_LIBRARY})


add_executable(main_glfw_opengl3
		main_glfw_opengl3.cpp
		bindings/imgui_impl_glfw.cpp
		bindings/imgui_impl_glfw.h
		bindings/imgui_impl_opengl3.cpp
		bindings/imgui_impl_opengl3.h
		)
target_link_libraries(main_glfw_opengl3 ${CONAN_LIBS} ${OPENGL_LIBRARY})


add_executable(show_image
		show_image.cpp
		bindings/imgui_impl_glfw.cpp
		bindings/imgui_impl_glfw.h
		bindings/imgui_impl_opengl3.cpp
		bindings/imgui_impl_opengl3.h
		)
target_link_libraries(show_image ${CONAN_LIBS} ${OPENGL_LIBRARY})

add_executable(main_glfw_opengl3_opengl
		main_glfw_opengl3_opengl.cpp
		bindings/imgui_impl_glfw.cpp
		bindings/imgui_impl_glfw.h
		bindings/imgui_impl_opengl3.cpp
		bindings/imgui_impl_opengl3.h
		linmath.h
		)
target_link_libraries(main_glfw_opengl3_opengl ${CONAN_LIBS} ${OPENGL_LIBRARY})
