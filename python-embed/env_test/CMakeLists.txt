cmake_minimum_required(VERSION 3.15)
project(python-embed)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(PYBIND11_PYTHON_VERSION 3.6)
set(PYBIND11_CPP_STANDARD -std=c++17)

find_package (Python3 COMPONENTS Interpreter Development)
find_package(pybind11)

include_directories(${PYTHON_INCLUDE_DIR})

foreach(FILE exec_numpy.py)
    file(COPY ${FILE} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endforeach()

foreach(APP test)
    add_executable(${APP} ${APP}.cpp)
    target_link_libraries(${APP} ${Python3_LIBRARY} pybind11::embed)
endforeach()
