cmake_minimum_required(VERSION 3.15)
project(flatcc_first C)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

#protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS sensor.proto sensor2.proto info.proto image.proto)

foreach(A ${CONAN_LIBS})
    message(STATUS ${A})
endforeach()

foreach(APP main test1)
    add_executable(${APP} ${APP}.c)
    target_link_libraries(${APP} PRIVATE ${CONAN_LIBS})
    target_include_directories(${APP} PUBLIC ${CMAKE_BINARY_DIR})
    #target_include_directories(${APP} PUBLIC ${CONAN_INCLUDE_DIRS_FLATCC})
    #set_target_properties(${APP} PROPERTIES CXX_STANDARD 17)
endforeach()


#message(STATUS "======================================================")
#get_cmake_property(_variableNames VARIABLES)
#list (SORT _variableNames)
#foreach (_variableName ${_variableNames})
#    message(STATUS "${_variableName}=${${_variableName}}")
#endforeach()