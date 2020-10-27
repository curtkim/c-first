cmake_minimum_required(VERSION 3.15)
project(torch-first LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#find_package(Torch REQUIRED PATHS "/home/curt/.conan/data/torch/1.6.0/curt/prebuilt/package/06cb18a658a30c9ee232f93db1f894f286607974")

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

#torch
#torch_library
#/usr/local/cuda/lib64/stubs/libcuda.so
#/usr/local/cuda/lib64/libnvrtc.so
#/usr/local/cuda/lib64/libnvToolsExt.so
#/usr/local/cuda/lib64/libcudart.so
#/home/curt/.conan/data/torch/1.6.0/curt/prebuilt/package/06cb18a658a30c9ee232f93db1f894f286607974/lib/libc10.so
#/home/curt/.conan/data/torch/1.6.0/curt/prebuilt/package/06cb18a658a30c9ee232f93db1f894f286607974/lib/libc10_cuda.so


foreach(APP
		04_cuda_detect
		)
	add_executable(${APP} ${APP}.cpp)
	target_link_directories(${APP} PUBLIC /usr/local/cuda/lib64)
	target_link_libraries(${APP} PUBLIC cudart nvToolsExt)
	#target_link_directories(${APP} PUBLIC /usr/local/cuda/lib64/stub)
	target_link_libraries(${APP} PUBLIC ${TORCH_LIBRARIES} ${CONAN_LIBS} c10 tensorpipe torch_cpu gomp torch_cuda torch c10_cuda )
	target_compile_definitions(${APP} PUBLIC WITH_CUDA)
	target_compile_features(${APP} PUBLIC cxx_std_17)
endforeach()
