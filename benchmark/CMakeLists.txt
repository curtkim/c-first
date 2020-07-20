cmake_minimum_required(VERSION 3.15)
project(benchmark)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

foreach(APP google nano_first nano_complexity_set nano_lodepng)
	add_executable(${APP} ${APP}.cpp)
	target_link_libraries(${APP} ${CONAN_LIBS})
	target_compile_features(${APP} PRIVATE cxx_std_17)
endforeach()
