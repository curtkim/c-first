cmake_minimum_required(VERSION 3.1.2)
project(conan_protobuf CXX)

set(CMAKE_VERBOSE_MAKEFILE ON)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup(TARGETS)

find_package(Protobuf REQUIRED)

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS sensor.proto sensor2.proto info.proto)
#protobuf_generate_python(PROTO_PYS sensor.proto)
#add_custom_target(proto_python ALL DEPENDS ${PROTO_PYS})

foreach(APP
        write
        read_by_sensor2
        write_like_apollo
        test_string_piece)
    add_executable(${APP} ${APP}.cc ${PROTO_SRCS} ${PROTO_HDRS})
    target_link_libraries(${APP} PRIVATE CONAN_PKG::protobuf)
    target_include_directories(${APP} PRIVATE ${CMAKE_BINARY_DIR})
    set_target_properties(${APP} PROPERTIES CXX_STANDARD 17)
endforeach()
