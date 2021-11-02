cmake_minimum_required(VERSION 3.15)
project(flatbuffer_first CXX C)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup(TARGETS)

#protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS sensor.proto sensor2.proto info.proto image.proto)

foreach(A ${CONAN_LIBS})
    message(STATUS ${A})
endforeach()

add_library(mallochook SHARED mallochook.c)

foreach(APP
        01_build_read
        02_minimal_test
        03_person_use
        04_json)
    add_executable(${APP} ${APP}.cpp)
    target_link_libraries(${APP} PRIVATE CONAN_PKG::flatbuffers mallochook dl)
    target_include_directories(${APP} PRIVATE ${CMAKE_BINARY_DIR})
    set_target_properties(${APP} PROPERTIES CXX_STANDARD 17)
endforeach()
