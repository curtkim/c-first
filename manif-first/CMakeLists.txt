cmake_minimum_required(VERSION 3.19)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(manif-first LANGUAGES CXX )

#set(CMAKE_VERBOSE_MAKEFILE ON)
#set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

foreach(APP 01_so2_average
		11_se2_average 12_se2_interpolation 13_se2_localization)
	add_executable(${APP} ${APP}.cpp)
	target_link_libraries(${APP} PUBLIC ${CONAN_LIBS})
	target_compile_features(${APP} PUBLIC cxx_std_17)
endforeach()
