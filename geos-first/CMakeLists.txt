cmake_minimum_required(VERSION 2.8.12)
project(geos-first)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_executable(line line.cpp)
target_link_libraries(line ${CONAN_LIBS})
target_compile_features(line PRIVATE cxx_std_17)
