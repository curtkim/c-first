cmake_minimum_required(VERSION 3.15)
project(torch-first LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CUDA_DYNAMIC_LIBRARY cuda nvrtc nvToolsExt cudart cufft curand cudnn cublas)
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


foreach(APP
		04_cuda_detect
		)
	add_executable(${APP} ${APP}.cpp)
	target_link_directories(${APP} PUBLIC /usr/local/cuda/lib64)
	target_link_libraries(${APP} PUBLIC ${CONAN_LIBS} ${CUDA_DYNAMIC_LIBRARY})
	target_compile_features(${APP} PUBLIC cxx_std_17)
	set_target_properties(${APP} PROPERTIES LINK_WHAT_YOU_USE "TRUE")
endforeach()
