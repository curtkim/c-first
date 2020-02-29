cmake_minimum_required (VERSION 3.2 FATAL_ERROR)
project (modern-cpp)

set (CMAKE_CXX_STANDARD_REQUIRED ON)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_executable(main main.cpp lib1.cpp lib1.h lib2.cpp lib2.h)
set_property(TARGET main PROPERTY CXX_STANDARD 17)
