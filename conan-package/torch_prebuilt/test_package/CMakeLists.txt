cmake_minimum_required(VERSION 3.15)
project(PackageTest CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

#foreach(ITEM ${CONAN_LIBS})
#	message( ${ITEM} )
#endforeach()

add_executable(example example.cpp)
target_link_directories(example PUBLIC /usr/local/cuda/lib64)
target_link_libraries(example ${CONAN_LIBS} cuda nvrtc nvToolsExt cudart cufft curand cudnn cublas)


# CTest is a testing tool that can be used to test your project.
# enable_testing()
# add_test(NAME example
#          WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
#          COMMAND example)
