cmake_minimum_required(VERSION 2.8.12)
project(conan-imgui)

find_package(OpenGL REQUIRED)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

foreach(FILE
		bunny.off)
	file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/${FILE} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)
endforeach()

foreach(APP
		02_binding_glfw_opengl3
		03_show_image
		05_custom_shape
		11_first
		12_model
		41_docking
		99_test
		)

	add_executable(${APP}
			${APP}.cpp
			bindings/imgui_impl_glfw.cpp
			bindings/imgui_impl_opengl3.cpp
			opengl_shader.cpp
			)
	target_compile_definitions(${APP} PUBLIC IMGUI_IMPL_OPENGL_LOADER_GLAD)
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
target_compile_definitions(21_moduled PUBLIC IMGUI_IMPL_OPENGL_LOADER_GLAD)
target_link_libraries(21_moduled ${CONAN_LIBS} ${OPENGL_LIBRARY} X11)
target_compile_features(21_moduled PUBLIC cxx_std_17)

