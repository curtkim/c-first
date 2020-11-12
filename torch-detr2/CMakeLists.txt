cmake_minimum_required(VERSION 3.15)
project(torch-detr2 LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CUDA_DYNAMIC_LIBRARY cuda nvrtc nvToolsExt cudart cufft curand cudnn cublas)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

message(STATUS "------------------")
foreach(A ${CONAN_LIBS})
	message(STATUS ${A})
endforeach()

foreach(APP main test)
	add_executable(${APP} ${APP}.cpp image_io.cpp)
	target_link_directories(${APP} PUBLIC /usr/local/cuda/lib64)
	target_link_libraries(${APP} PUBLIC ${CONAN_LIBS} ${CUDA_DYNAMIC_LIBRARY})
	target_compile_features(${APP} PUBLIC cxx_std_17)
	set_target_properties(${APP} PROPERTIES LINK_WHAT_YOU_USE "TRUE")
endforeach()


add_library(carla_common OBJECT carla_common.cpp)
add_library(viz_opengl OBJECT viz_opengl.cpp)
add_library(detr OBJECT detr.cpp)
target_compile_features(detr PUBLIC cxx_std_17)

foreach(APP pipeline1)
	add_executable(${APP} ${APP}.cpp)
	target_include_directories(${APP} PUBLIC /usr/local/cuda/include)
	target_link_directories(${APP} PUBLIC /usr/local/cuda/lib64)
	target_link_libraries(${APP} PUBLIC ${CONAN_LIBS} ${OPENGL_LIBRARY} ${CUDA_DYNAMIC_LIBRARY} X11 carla_common viz_opengl detr)
	target_compile_features(${APP} PUBLIC cxx_std_17)
	set_target_properties(${APP} PROPERTIES LINK_WHAT_YOU_USE "TRUE")
endforeach()
