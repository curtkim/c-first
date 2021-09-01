cmake_minimum_required(VERSION 3.15)
project(conan-imgui)

find_package(OpenGL REQUIRED)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

foreach(APP
		00_timer
		01_first
		)

	add_executable(${APP}
			${APP}.cpp
			bindings/imgui_impl_glfw.cpp
			bindings/imgui_impl_opengl3.cpp
			)
	target_compile_definitions(${APP} PUBLIC IMGUI_IMPL_OPENGL_LOADER_GLAD)
	target_link_libraries(${APP} ${CONAN_LIBS} ${OPENGL_LIBRARY} X11 pthread)

	target_compile_options(${APP} PRIVATE -fcoroutines)
	target_compile_features(${APP} PUBLIC cxx_std_20)
	target_compile_definitions(${APP} PUBLIC ASIO_HAS_STD_COROUTINE)
	target_compile_definitions(${APP} PUBLIC ASIO_HAS_CO_AWAIT)

endforeach()
