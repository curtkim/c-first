cmake_minimum_required(VERSION 3.16)
project(protobuf3-first CXX)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
#conan_basic_setup(TARGETS) # protobuf include가 포함이 안된다.
conan_basic_setup()

protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS sample.proto)

foreach(APP
        01_write)
    add_executable(${APP} ${APP}.cpp ${PROTO_SRCS} ${PROTO_HDRS})
    target_include_directories(${APP} PRIVATE ${CMAKE_BINARY_DIR})
    target_link_libraries(${APP} PRIVATE ${CONAN_LIBS})
    set_target_properties(${APP} PROPERTIES CXX_STANDARD 17)
endforeach()
