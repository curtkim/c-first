cmake_minimum_required(VERSION 3.15)
project(PackageTest CXX)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

find_package(Threads REQUIRED)

add_executable(example example.cpp)
target_link_libraries(example ${CONAN_LIBS} X11)
target_compile_features(example PRIVATE cxx_std_14)
