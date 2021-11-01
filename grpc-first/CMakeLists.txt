cmake_minimum_required(VERSION 3.15)
project(grpc-first)

set ( CMAKE_EXPORT_COMPILE_COMMANDS on )

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

set(CMAKE_PREFIX_PATH ${CMAKE_BINARY_DIR})
set(CMAKE_MODULE_PATH ${CMAKE_BINARY_DIR})

# Find Protobuf installation

set(protobuf_MODULE_COMPATIBLE TRUE)
find_package(protobuf CONFIG REQUIRED)
message(STATUS "Using protobuf ${protobuf_VERSION}")
if (CMAKE_CROSSCOMPILING)
  find_program(Protoc_Executable protoc PATHS ENV PATH NO_DEFAULT_PATH)
else()
  set(Protoc_Executable ${Protobuf_PROTOC_EXECUTABLE})
endif()

# Find gRPC installation
find_package(gRPC CONFIG REQUIRED)
message(STATUS "Using gRPC ${gRPC_VERSION}")
set(_GRPC_CPP_PLUGIN_EXECUTABLE $<TARGET_FILE:gRPC::grpc_cpp_plugin>)


# Proto file
get_filename_component(hw_proto "helloworld.proto" ABSOLUTE)
get_filename_component(hw_proto_path "${hw_proto}" PATH)

# Generated sources
set(hw_proto_srcs "${CMAKE_CURRENT_BINARY_DIR}/helloworld.pb.cc")
set(hw_proto_hdrs "${CMAKE_CURRENT_BINARY_DIR}/helloworld.pb.h")
set(hw_grpc_srcs "${CMAKE_CURRENT_BINARY_DIR}/helloworld.grpc.pb.cc")
set(hw_grpc_hdrs "${CMAKE_CURRENT_BINARY_DIR}/helloworld.grpc.pb.h")
add_custom_command(
      OUTPUT "${hw_proto_srcs}" "${hw_proto_hdrs}" "${hw_grpc_srcs}" "${hw_grpc_hdrs}"
      COMMAND ${Protoc_Executable}
      ARGS --grpc_out "${CMAKE_CURRENT_BINARY_DIR}"
        --cpp_out "${CMAKE_CURRENT_BINARY_DIR}"
        -I "${hw_proto_path}"
        --plugin=protoc-gen-grpc="${_GRPC_CPP_PLUGIN_EXECUTABLE}"
        "${hw_proto}"
      DEPENDS "${hw_proto}")

# Build generated sources as an object library
add_library(hw_proto_obj OBJECT ${hw_proto_srcs} ${hw_grpc_srcs})
target_compile_features(hw_proto_obj PUBLIC cxx_std_11)

# Include generated *.pb.h files
include_directories("${CMAKE_CURRENT_BINARY_DIR}")


foreach(APP
        greeter_client_server
        greeter_async_client
        greeter_async_server
        greeter_async_server2
        )
    add_executable(${APP} ${APP}.cpp)
    target_compile_features(${APP} PUBLIC cxx_std_17)
    target_link_libraries(${APP} PUBLIC ${CONAN_LIBS} hw_proto_obj)
endforeach()
