cmake_minimum_required(VERSION 3.15)
project(python-embed)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

set(PYBIND11_PYTHON_VERSION 3.7)
set(PYBIND11_CPP_STANDARD -std=c++17)

find_package (Python3 COMPONENTS Interpreter Development)

#get_cmake_property(_variableNames VARIABLES)
#list (SORT _variableNames)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()

include_directories(${PYTHON_INCLUDE_DIR})

foreach(FILE calc.py calc_global.py img.py exec_numpy.py)
    file(COPY ${FILE} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)
endforeach()


foreach(APP 01_execute_script 02_module 03_embeded_module 04_mat_to_python 05_min_rosen)
    add_executable(${APP} ${APP}.cpp)
    target_link_libraries(${APP} ${Python3_LIBRARY} ${CONAN_LIBS} pthread dl util)
endforeach()

