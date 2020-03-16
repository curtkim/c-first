cmake_minimum_required(VERSION 2.8.12)
project(myproject)

add_definitions("-std=c++17")

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_executable(md5 src/md5.cpp src/transport/car.cpp src/ExampleClass.cpp)
target_link_libraries(md5 ${CONAN_LIBS})
