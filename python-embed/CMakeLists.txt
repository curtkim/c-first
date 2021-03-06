cmake_minimum_required(VERSION 3.10)
project(python-embed)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(PYBIND11_PYTHON_VERSION 3.8)
set(PYBIND11_CPP_STANDARD -std=c++17)

find_package (Python3 COMPONENTS Interpreter Development)

#find_package(pybind11)
#get_cmake_property(_variableNames VARIABLES)
#list (SORT _variableNames)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()

include_directories(${PYTHON_INCLUDE_DIR})

foreach(FILE calc.py calc_global.py myimg.py exec_numpy.py numpyfuncs.py)
    file(COPY ${FILE} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)
endforeach()

foreach(APP 01_execute_script 02_module 03_embeded_module 04_mat_to_python 05_min_rosen 06_eigen)
    add_executable(${APP} ${APP}.cpp)
    target_link_libraries(${APP} ${Python3_LIBRARY} ${CONAN_LIBS} pthread dl util)
endforeach()

