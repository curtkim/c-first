cmake_minimum_required(VERSION 3.19)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(taskflow-first LANGUAGES CUDA CXX )


#set(CMAKE_VERBOSE_MAKEFILE ON)
#set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CUDA_DYNAMIC_LIBRARY cuda nvrtc nvToolsExt cudart cufft curand cudnn cublas)

message(STATUS ${CONAN_LIBS})

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

foreach(APP 01_first 02_subflow 03_cond)
	add_executable(${APP} ${APP}.cpp)
	target_link_libraries(${APP} PUBLIC ${CONAN_LIBS})
	target_compile_features(${APP} PUBLIC cxx_std_17)
endforeach()

foreach(APP 51_cuda-first 52_saxpy)
	add_executable(${APP} ${APP}.cu)
	target_include_directories(${APP} PRIVATE /usr/local/cuda/include)
	target_link_directories(${APP} PUBLIC /usr/local/cuda/lib64)
	target_link_libraries(${APP} PUBLIC ${CONAN_LIBS} ${CUDA_DYNAMIC_LIBRARY} )
	target_compile_features(${APP} PUBLIC cxx_std_17)
	set_target_properties(${APP} PROPERTIES LINK_WHAT_YOU_USE "TRUE")

	if(NOT CUDA_ARCHITECTURES)
		set_property(TARGET ${APP} PROPERTY CUDA_ARCHITECTURES OFF)
	endif()

	target_compile_options(${APP} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
			--expt-relaxed-constexpr
			--expt-extended-lambda
			--default-stream per-thread
			--use_fast_math
			-Xcudafe "--diag_suppress=20011 --diag_suppress=20014"
			>)
	#-Xcudafe --display_error_number
	# 를 추가해서 error no를 보고
	# --diag_suppress에 추가한다.
endforeach()

