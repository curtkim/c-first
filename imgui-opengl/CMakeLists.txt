cmake_minimum_required(VERSION 3.15)
project(imgui-opengl)

find_package(OpenGL REQUIRED)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

#foreach(FILE bunny.off)
#	file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/${FILE} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)
#endforeach()



add_executable(main
		source/main.cpp
		source/application.cpp
		source/window/jgl_window.cpp
		source/shader/shader_util.cpp
		source/ui/property_panel.cpp
		source/ui/scene_view.cpp
		source/render/opengl_buffer_manager.cpp
		source/render/opengl_context.cpp
		source/render/ui_context.cpp
		source/mesh_import/import_obj.cpp
		source/elems/mesh.cpp

		bindings/imgui_impl_glfw.cpp
		bindings/imgui_impl_opengl3.cpp
		)
target_include_directories(main PUBLIC source)
target_include_directories(main PUBLIC bindings)
target_compile_definitions(main PUBLIC IMGUI_IMPL_OPENGL_LOADER_GLAD)
target_compile_definitions(main PUBLIC IMGUI_HAS_DOCK)
target_link_libraries(main ${CONAN_LIBS} ${OPENGL_LIBRARY} X11)
target_compile_features(main PUBLIC cxx_std_17)
