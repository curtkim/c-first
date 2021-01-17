cmake_minimum_required(VERSION 3.15)
project(range-first)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

foreach(APP views algorithm split_string)
    add_executable(${APP} ${APP}.cpp)
    target_compile_features(${APP} PUBLIC cxx_std_17)
endforeach()

add_executable(helloworld helloworld.cpp)
target_compile_features(helloworld PUBLIC cxx_std_20)
