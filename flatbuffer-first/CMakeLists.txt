cmake_minimum_required(VERSION 3.15)
project(flatbuffer_first CXX)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup(TARGETS)

#protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS sensor.proto sensor2.proto info.proto image.proto)

foreach(A ${CONAN_LIBS})
    message(STATUS ${A})
endforeach()

foreach(APP
        main)
    add_executable(${APP} ${APP}.cpp)
    target_link_libraries(${APP} PRIVATE CONAN_PKG::flatbuffers)
    target_include_directories(${APP} PRIVATE ${CMAKE_BINARY_DIR})
    set_target_properties(${APP} PROPERTIES CXX_STANDARD 17)
endforeach()
