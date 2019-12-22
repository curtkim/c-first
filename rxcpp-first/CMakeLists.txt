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
add_executable(complex complex.cpp)
set_property(TARGET complex PROPERTY CXX_STANDARD 17)
target_link_libraries (complex rxcpp)

add_executable(create create.cpp)
set_property(TARGET create PROPERTY CXX_STANDARD 17)
target_link_libraries (create rxcpp)

add_executable(op op.cpp)
set_property(TARGET op PROPERTY CXX_STANDARD 17)
target_link_libraries (op rxcpp)

add_executable(schedulers schedulers.cpp)
set_property(TARGET schedulers PROPERTY CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wformat=0")
target_link_libraries (schedulers rxcpp pthread)

add_executable(read_file read_file.cpp)
set_property(TARGET read_file PROPERTY CXX_STANDARD 17)
target_link_libraries (read_file rxcpp pthread)

add_executable(coordination coordination.cpp)
set_property(TARGET coordination PROPERTY CXX_STANDARD 17)
target_link_libraries (coordination rxcpp pthread)
