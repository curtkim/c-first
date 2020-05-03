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
target_link_libraries(main_glfw_opengl2 ${CONAN_LIBS} ${OPENGL_LIBRARY} X11)


foreach(APP
		main_glfw_opengl3
		main_glfw_opengl3_opengl
		show_image)

	add_executable(${APP}
			${APP}.cpp
			bindings/imgui_impl_glfw.cpp
			bindings/imgui_impl_glfw.h
			bindings/imgui_impl_opengl3.cpp
			bindings/imgui_impl_opengl3.h
			)
	target_compile_definitions(${APP} PUBLIC STB_IMAGE_IMPLEMENTATION)
	target_link_libraries(${APP} ${CONAN_LIBS} ${OPENGL_LIBRARY} X11)
endforeach()


add_executable(main
		main.cpp
		bindings/imgui_impl_glfw.cpp
		bindings/imgui_impl_glfw.h
		bindings/imgui_impl_opengl3.cpp
		bindings/imgui_impl_opengl3.h
		opengl_shader.cpp
		opengl_shader.h
		)
target_link_libraries(main ${CONAN_LIBS} ${OPENGL_LIBRARY} X11)
