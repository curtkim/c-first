cmake_minimum_required(VERSION 3.15)
project(torch-first LANGUAGES CUDA CXX)

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
		02_torch_first
		)
	add_executable(${APP} ${APP}.cpp)
	target_link_libraries(${APP} PUBLIC ${CONAN_LIBS})
	target_compile_features(${APP} PUBLIC cxx_std_17)
endforeach()

# CUDA
foreach(APP
		04_cuda_detect
		91_candy
		)
	add_executable(${APP} ${APP}.cpp)
	target_link_directories(${APP} PUBLIC /usr/local/cuda/lib64)
	target_link_libraries(${APP} PUBLIC ${CONAN_LIBS} ${CUDA_DYNAMIC_LIBRARY} image_io)
	target_compile_features(${APP} PUBLIC cxx_std_17)
	set_target_properties(${APP} PROPERTIES LINK_WHAT_YOU_USE "TRUE")
endforeach()


add_executable(11_cuda_accessor 11_cuda_accessor.cu)
target_link_directories(11_cuda_accessor PUBLIC /usr/local/cuda/lib64)
target_link_libraries(11_cuda_accessor PUBLIC ${CONAN_LIBS} ${CUDA_DYNAMIC_LIBRARY})
target_compile_features(11_cuda_accessor PUBLIC cxx_std_17)
set_target_properties(11_cuda_accessor PROPERTIES LINK_WHAT_YOU_USE "TRUE")
