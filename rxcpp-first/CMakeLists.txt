cmake_minimum_required (VERSION 3.2 FATAL_ERROR)
project (rxcpp-first)

set (CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(rxcpp CONFIG)

#get_cmake_property(_variableNames VARIABLES)
#list (SORT _variableNames)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()

# add the executable
add_executable(rxcpp-first main.cpp)
set_property(TARGET rxcpp-first PROPERTY CXX_STANDARD 17)
target_link_libraries (rxcpp-first rxcpp)

add_executable(rxcpp-main2 main2.cpp)
set_property(TARGET rxcpp-main2 PROPERTY CXX_STANDARD 17)
target_link_libraries (rxcpp-main2 rxcpp)

add_executable(rxcpp-main3 main3.cpp)
set_property(TARGET rxcpp-main3 PROPERTY CXX_STANDARD 17)
target_link_libraries (rxcpp-main3 rxcpp)
