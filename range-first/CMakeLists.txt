cmake_minimum_required(VERSION 3.15)
project(range-first)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_executable(views views.cpp)
target_link_libraries(views) # header only library
target_compile_features(views PUBLIC cxx_std_17)

add_executable(algorithm algorithm.cpp)
target_link_libraries(algorithm)
target_compile_features(algorithm PUBLIC cxx_std_17)

add_executable(helloworld helloworld.cpp)
target_link_libraries(helloworld)
target_compile_features(helloworld PUBLIC cxx_std_20)
