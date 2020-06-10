cmake_minimum_required(VERSION 3.15)
project(conan-imgui)

find_package(OpenGL REQUIRED)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_executable(main
		main.cpp
		gui.cpp
		bindings/imgui_impl_glfw.cpp
		bindings/imgui_impl_opengl3.cpp
		)
target_compile_definitions(main PUBLIC IMGUI_IMPL_OPENGL_LOADER_GLAD)
target_link_libraries(main ${CONAN_LIBS} ${OPENGL_LIBRARY} X11)
target_compile_features(main PUBLIC cxx_std_17)
