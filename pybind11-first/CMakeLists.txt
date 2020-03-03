cmake_minimum_required(VERSION 2.8.12)
project(MD5Encrypter)

#add_definitions("-std=c++11")

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

# Using the pybind cmake helpers:
set(PYBIND11_PYTHON_VERSION 3.7)
find_package(pybind11 REQUIRED)

pybind11_add_module(vector vector.cpp)
pybind11_add_module(numpy1d numpy1d.cpp)
pybind11_add_module(numpy2d numpy2d.cpp)
pybind11_add_module(numpy2d_eigin numpy2d_eigen.cpp)
