cmake_minimum_required(VERSION 3.15)
project(pybind11-first)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

# Using the pybind cmake helpers:
find_package(pybind11 REQUIRED)

set(PYBIND11_PYTHON_VERSION 3.7)
set(PYBIND11_CPP_STANDARD -std=c++17)

add_library(vector_inner vector_inner.cpp)
set_property(TARGET vector_inner PROPERTY POSITION_INDEPENDENT_CODE ON) # -fPIC

pybind11_add_module(vector vector.cpp)
target_link_libraries (vector PRIVATE vector_inner)

pybind11_add_module(numpy1d numpy1d.cpp)
pybind11_add_module(numpy2d numpy2d.cpp)
pybind11_add_module(numpy2d_eigin numpy2d_eigen.cpp)


pybind11_add_module(oop oop/oop_pybind.cpp oop/person.cpp)
target_include_directories(oop PRIVATE .)

pybind11_add_module(nested nested/nested_pybind.cpp nested/util/format.cpp)
target_include_directories(nested PRIVATE .)
#set_property(TARGET nested PROPERTY POSITION_INDEPENDENT_CODE ON) # -fPIC