cmake_minimum_required(VERSION 3.18)
project(PackageTest CXX)

#set(CMAKE_VERBOSE_MAKEFILE ON)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/bike_duck.jpg
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)

add_executable(example example.cpp)
target_link_libraries(example ${CONAN_LIBS})
target_compile_features(example PRIVATE cxx_std_17)
