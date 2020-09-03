cmake_minimum_required (VERSION 3.2 FATAL_ERROR)
project (sys)

set (CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(producer producer.cpp)
add_executable(consumer consumer.cpp)

add_executable(graceful graceful.cpp)
target_link_libraries( graceful pthread )

add_executable(server server.cpp)
add_executable(client client.cpp)

add_executable(namedpipe_write namedpipe_write.cpp)
add_executable(namedpipe_read namedpipe_read.cpp)

add_executable(memory_mapped_write memory_mapped_write.cpp)
add_executable(memory_mapped_read memory_mapped_read.cpp)
add_executable(mmap_test mmap_test.cpp)

add_executable(inotify_first inotify_first.cpp)
add_executable(write_direct write_direct.cpp)

add_executable(epoll_echo_server epoll_echo_server.cpp)

#
add_executable(test_read_ifstream test_read_ifstream.cpp)
add_executable(test_read_mmap test_read_mmap.cpp)

add_executable(test_aio_passwd test_aio_passwd.c)
add_executable(test_aio_poll test_aio_poll.c)
