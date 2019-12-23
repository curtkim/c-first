cmake_minimum_required (VERSION 3.2 FATAL_ERROR)
project (rxcpp-first)

set (CMAKE_CXX_STANDARD_REQUIRED ON)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

#get_cmake_property(_variableNames VARIABLES)
#list (SORT _variableNames)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()

# add the executable
add_executable(complex complex.cpp)
set_property(TARGET complex PROPERTY CXX_STANDARD 17)
target_link_libraries (complex ${CONAN_LIBS})

add_executable(create create.cpp)
set_property(TARGET create PROPERTY CXX_STANDARD 17)
target_link_libraries (create ${CONAN_LIBS})

add_executable(op op.cpp)
set_property(TARGET op PROPERTY CXX_STANDARD 17)
target_link_libraries (op ${CONAN_LIBS})

add_executable(schedulers schedulers.cpp)
set_property(TARGET schedulers PROPERTY CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wformat=0")
target_link_libraries (schedulers ${CONAN_LIBS} pthread)

add_executable(read_file read_file.cpp)
set_property(TARGET read_file PROPERTY CXX_STANDARD 17)
target_link_libraries (read_file ${CONAN_LIBS} pthread)

add_executable(coordination coordination.cpp)
set_property(TARGET coordination PROPERTY CXX_STANDARD 17)
target_link_libraries (coordination ${CONAN_LIBS} pthread)

#add_executable(rxcurl_ex rxcurl_ex.cpp)
#set_property(TARGET rxcurl_ex PROPERTY CXX_STANDARD 17)
#target_link_libraries (rxcurl_ex ${CONAN_LIBS} pthread)

add_executable(omega omega.cpp)
set_property(TARGET omega PROPERTY CXX_STANDARD 17)
target_link_libraries (omega ${CONAN_LIBS} pthread)

add_executable(async async.cpp)
set_property(TARGET async PROPERTY CXX_STANDARD 17)
target_link_libraries (async ${CONAN_LIBS} pthread)

add_executable(rxrest rxrest.cpp)
set_property(TARGET rxrest PROPERTY CXX_STANDARD 17)
target_link_libraries (rxrest ${CONAN_LIBS} pthread)
