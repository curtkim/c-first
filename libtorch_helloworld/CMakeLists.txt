cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
project(dcgan)

find_package(Torch REQUIRED)

add_executable(dcgan dcgan.cpp)
target_link_libraries(dcgan "${TORCH_LIBRARIES}")
set_property(TARGET dcgan PROPERTY CXX_STANDARD 17)

add_executable(aten-first aten-first.cpp)
target_link_libraries(aten-first "${TORCH_LIBRARIES}")
set_property(TARGET aten-first PROPERTY CXX_STANDARD 17)