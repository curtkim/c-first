cmake_minimum_required(VERSION 2.8.12)
project(conan-imgui)

add_definitions("-std=c++17")

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

foreach(APP aten_first autograd_first dcgan torchscript)
	add_executable(${APP} ${APP}.cpp)
	target_link_libraries(${APP} ${CONAN_LIBS})
endforeach()
