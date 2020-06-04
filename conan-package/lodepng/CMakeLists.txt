cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(lodepng VERSION 1.0 LANGUAGES CXX)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

option(BUILD_EXAMPLES "Build the example applications" OFF)

add_library(lodepng STATIC lodepng/lodepng.cpp)
install(TARGETS lodepng DESTINATION lib)
install(FILES lodepng.h DESTINATION include)

if (BUILD_EXAMPLES)
  add_subdirectory(examples)
endif (BUILD_EXAMPLES)