cmake_minimum_required(VERSION 3.15)
project(asio)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

# TODO
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
#add_executable(range_based_for range_based_for.cpp)
#target_compile_features(range_based_for PUBLIC cxx_std_20)
#target_compile_options(range_based_for PRIVATE -fcoroutines-ts)
#target_compile_definitions(range_based_for PRIVATE ASIO_HAS_CO_AWAIT)
#target_link_libraries(range_based_for ${CONAN_LIBS})

foreach(APP
    01_timer_interval
    01_timer_system
    02_signal
    11_daytime_client
    21_stdin_read 21_stdin_read_no_thread 22_stdout_write
    23_pipe_stdout 23_pipe_stdin
    29_alarm
    31_file_write_c 32_file_write_future 33_file_watch
    41_async_tcp_echo_server 41_async_tcp_echo_server_wo_class 42_tcp_echo_client
    61_domain_echo_server 62_domain_echo_client
    63_domain_server_sync 64_domain_client_sync
    65_unnamed_socket
    71_chat_server 72_chat_client
    91_server
    )
  add_executable(${APP} ${APP}.cpp)
  target_link_libraries(${APP} ${CONAN_LIBS} stdc++fs)
  #target_compile_definitions(${APP} PUBLIC BOOST_ASIO_DISABLE_EPOLL)
  target_compile_features(${APP} PUBLIC cxx_std_17)
endforeach()

## fixed_size_header
add_executable(fixed_size_header_client fixed_size_header/client.cpp)
target_link_libraries(fixed_size_header_client ${CONAN_LIBS})

add_executable(fixed_size_header_server fixed_size_header/server.cpp)
target_link_libraries(fixed_size_header_server ${CONAN_LIBS})

## http_client
add_executable(http_client http_client/http_client.cpp http_client/AsyncHttpClient.cpp)
target_link_libraries(http_client ${CONAN_LIBS})
target_compile_features(http_client PRIVATE cxx_std_17)

## protobuf

#  ~/.conan/data/protoc_installer/3.9.1/bincrafters/stable/package/c0c1ef10e3d0ded44179e28b669d6aed0277ca6a/bin/protoc --cpp_out=. stringdb.proto
add_executable(protobuf_server protobuf/server_main.cpp protobuf/db_server.cpp protobuf/stringdb.pb.cc)
target_link_libraries(protobuf_server ${CONAN_LIBS})
target_compile_features(protobuf_server PRIVATE cxx_std_17)
