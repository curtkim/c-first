cmake_minimum_required(VERSION 3.15)
project(torch-first LANGUAGES CUDA CXX)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

include_directories(/usr/local/cuda-10.1/include)

message(STATUS "------------------")
foreach(A ${CONAN_LIBS})
	message(STATUS ${A})
endforeach()

foreach(APP
		00_aten
		01_aten_first
		02_first
		03_autograd_first
		04_cuda_detect
		21_torchscript
		31_input
		#41_inference
		#42_inference
		43_inference
		51_TwoLayerNetwork 52_dcgan
		)
	add_executable(${APP} ${APP}.cpp)
	target_link_libraries(${APP} ${TORCH_LIBRARIES} ${CONAN_LIBS})
	target_compile_definitions(${APP} PUBLIC WITH_CUDA)
	target_compile_features(${APP} PUBLIC cxx_std_17)
endforeach()
