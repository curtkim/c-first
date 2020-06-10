cmake_minimum_required(VERSION 2.8.12)
project(conan-imgui)

# opengl loader를 GLAD로 설정한다.
add_definitions(-DIMGUI_IMPL_OPENGL_LOADER_GLAD)

find_package(OpenGL REQUIRED)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()


foreach(APP
		01_first
		02_binding_glfw_opengl3
		03_show_image
		04_with_opengl)

	add_executable(${APP}
			${APP}.cpp
			bindings/imgui_impl_glfw.cpp
			bindings/imgui_impl_glfw.h
			bindings/imgui_impl_opengl3.cpp
			bindings/imgui_impl_opengl3.h
			opengl_shader.cpp
			opengl_shader.h
			)
	target_compile_definitions(${APP} PUBLIC STB_IMAGE_IMPLEMENTATION)
	target_link_libraries(${APP} ${CONAN_LIBS} ${OPENGL_LIBRARY} X11)
	target_compile_features(${APP} PUBLIC cxx_std_17)
endforeach()

add_executable(21_moduled
		21_moduled.cpp
		21_moduled_ui.cpp
		bindings/imgui_impl_glfw.cpp
		bindings/imgui_impl_opengl3.cpp
		)
target_compile_definitions(21_moduled PUBLIC STB_IMAGE_IMPLEMENTATION)
target_link_libraries(21_moduled ${CONAN_LIBS} ${OPENGL_LIBRARY} X11)
target_compile_features(21_moduled PUBLIC cxx_std_17)

