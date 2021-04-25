cmake_minimum_required(VERSION 3.16)
project(torch-first LANGUAGES CUDA CXX)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CUDA_DYNAMIC_LIBRARY cuda nvrtc nvToolsExt cudart cufft curand cudnn cublas)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

#include_directories(/usr/local/cuda/include)

message(STATUS "------------------")
foreach(A ${CONAN_LIBS})
	message(STATUS ${A})
endforeach()

message(STATUS "------------------")
foreach(ITEM ${TORCH_LIBRARIES})
	message( ${ITEM} )
endforeach()


file(COPY resources DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)

add_library(image_io OBJECT utils/image_io.cpp)

foreach(APP
		00_aten
		01_aten_named
		05_tensor_create_from_gpu_memory
		)
	add_executable(${APP} ${APP}.cpp)
	target_include_directories(${APP} PRIVATE /usr/local/cuda/include)
	target_link_directories(${APP} PUBLIC /usr/local/cuda/lib64)
	target_link_libraries(${APP} PUBLIC ${CONAN_LIBS} ${CUDA_DYNAMIC_LIBRARY})
	target_compile_features(${APP} PUBLIC cxx_std_17)
	set_target_properties(${APP} PROPERTIES LINK_WHAT_YOU_USE "TRUE")
endforeach()


add_executable(02_torch_first 02_torch_first.cpp)
target_precompile_headers(02_torch_first PRIVATE precompile.hpp)
target_include_directories(02_torch_first PRIVATE /usr/local/cuda/include)
target_link_directories(02_torch_first PUBLIC /usr/local/cuda/lib64)
target_link_libraries(02_torch_first PUBLIC ${CONAN_LIBS} ${CUDA_DYNAMIC_LIBRARY})
target_compile_features(02_torch_first PUBLIC cxx_std_17)
set_target_properties(02_torch_first PROPERTIES LINK_WHAT_YOU_USE "TRUE")


# CUDA
foreach(APP
		04_cuda_detect
		21_torchscript
		91_candy
		)
	add_executable(${APP} ${APP}.cpp)
	target_include_directories(${APP} PRIVATE /usr/local/cuda/include)
	target_link_directories(${APP} PUBLIC /usr/local/cuda/lib64)
	target_link_libraries(${APP} PUBLIC ${CONAN_LIBS} ${CUDA_DYNAMIC_LIBRARY} image_io)
	target_compile_features(${APP} PUBLIC cxx_std_17)
	set_target_properties(${APP} PROPERTIES LINK_WHAT_YOU_USE "TRUE")
endforeach()

foreach(APP 11_cuda_accessor 12_cuda_modify)
	add_executable(${APP} ${APP}.cu)
	target_link_directories(${APP} PUBLIC /usr/local/cuda/lib64)
	target_link_libraries(${APP} PUBLIC ${CONAN_LIBS} ${CUDA_DYNAMIC_LIBRARY})
	target_compile_features(${APP} PUBLIC cxx_std_17)
	set_target_properties(${APP} PROPERTIES LINK_WHAT_YOU_USE "TRUE")

	set_target_properties(${APP} PROPERTIES CUDA_ARCHITECTURES 61)
endforeach()