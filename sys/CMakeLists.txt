cmake_minimum_required (VERSION 3.2 FATAL_ERROR)
project (sys)

set (CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(01_producer 01_producer.cpp)
add_executable(01_consumer 01_consumer.cpp)

add_executable(02_graceful 02_signal_graceful.cpp)
target_link_libraries( 02_graceful pthread )

add_executable(03_socket_server 03_socket_server.cpp)
add_executable(03_socket_client 03_socket_client.cpp)
add_executable(03_epoll_echo_server 03_epoll_echo_server.cpp)


add_executable(04_namedpipe_write 04_namedpipe_write.cpp)
add_executable(04_namedpipe_read 04_namedpipe_read.cpp)
add_executable(04_unnamed_pipe 04_unnamed_pipe.cpp)

add_executable(05_memory_mapped_write 05_memory_mapped_write.cpp)
add_executable(05_memory_mapped_read 05_memory_mapped_read.cpp)
add_executable(05_mmap_test 05_mmap_test.cpp)

add_executable(06_inotify_first 06_inotify_first.cpp)

add_executable(07_write_direct 07_write_direct.cpp)
add_executable(07_write_direct2 07_write_direct2.cpp)

add_executable(08_read_ifstream 08_read_ifstream.cpp)
add_executable(08_read_mmap 08_read_mmap.cpp)

add_executable(09_aio_passwd 09_aio_passwd.c)
add_executable(09_aio_poll 09_aio_poll.c)

foreach(APP 11_poll 12_poll_stdin 14_select)
    add_executable(${APP} ${APP}.cpp)
    target_compile_features(${APP} PUBLIC cxx_std_17)
endforeach()
