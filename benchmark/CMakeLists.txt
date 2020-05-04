cmake_minimum_required(VERSION 2.8.12)
project(benchmark)

#add_definitions("-std=c++17")

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

foreach(APP first)
	add_executable(${APP} ${APP}.cpp)
	target_link_libraries(${APP} ${CONAN_LIBS})
	target_compile_features(${APP} PRIVATE cxx_std_17)
endforeach()
