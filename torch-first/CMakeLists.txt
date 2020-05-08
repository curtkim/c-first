cmake_minimum_required(VERSION 3.10)
project(conan-imgui)

add_definitions("-std=c++17")

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

find_package(Torch REQUIRED)

#link_directories( /home/curt/.pyenv/versions/3.7.5/lib/python3.7/site-packages/torch/lib)
#include_directories( /home/curt/.pyenv/versions/3.7.5/lib/python3.7/site-packages/torch/include)

#link_directories( /opt/conda/lib/python3.7/site-packages/torch/lib)
#include_directories( /opt/conda/lib/python3.7/site-packages/torch/include)

foreach(APP
		01_aten_first
		02_first
		03_autograd_first
		21_torchscript
		31_input
		41_inference
		51_TwoLayerNetwork 52_dcgan
		)
	add_executable(${APP} ${APP}.cpp)
	target_link_libraries(${APP} ${TORCH_LIBRARIES} ${CONAN_LIBS})
endforeach()
