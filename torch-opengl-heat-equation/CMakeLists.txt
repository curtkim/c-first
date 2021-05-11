cmake_minimum_required(VERSION 3.16)
project(torch-opengl-heat-equation LANGUAGES CUDA CXX)

find_package(OpenGL REQUIRED)

#set(CMAKE_VERBOSE_MAKEFILE ON)
#set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CUDA_DYNAMIC_LIBRARY cuda nvrtc nvToolsExt cudart cufft curand cudnn cublas)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

# CUDA
foreach(APP main)
	add_executable(${APP} ${APP}.cpp)
	target_include_directories(${APP} PRIVATE /usr/local/cuda/include)
	target_link_directories(${APP} PUBLIC /usr/local/cuda/lib64)
	target_link_libraries(${APP} PUBLIC ${CONAN_LIBS} ${CUDA_DYNAMIC_LIBRARY} EGL ${OPENGL_LIBRARY})
	target_compile_features(${APP} PUBLIC cxx_std_17)
	set_target_properties(${APP} PROPERTIES LINK_WHAT_YOU_USE "TRUE")
	target_compile_definitions(${APP} PUBLIC STB_IMAGE_IMPLEMENTATION)
endforeach()
