cmake_minimum_required(VERSION 2.8.12)
project(boost-first-1-58)

add_definitions("-std=c++11")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wformat=0")

find_package(Boost 1.58 REQUIRED system thread filesystem)

add_executable(thread_pool2 thread_pool2.cpp)
target_link_libraries(thread_pool2 ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY})

add_executable(asio_tutorial asio_01_single_thread_run_and_poll.cpp)
target_link_libraries(asio_tutorial ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY})

add_executable(asio_multi_thread_tutorial asio_02_multi_thread_global.cpp)
target_link_libraries(asio_multi_thread_tutorial ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY})

add_executable(asio_multi_thread_tutorial2 asio_03_multi_thread.cpp)
target_link_libraries(asio_multi_thread_tutorial2 ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY})

add_executable(asio_multi_thread_tutorial_lock asio_04_multi_thread_tutorial_lock.cpp)
target_link_libraries(asio_multi_thread_tutorial_lock ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY})

add_executable(asio_post asio_31_post.cpp)
target_link_libraries(asio_post ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY})

add_executable(asio_dispatch asio_32_dispatch.cpp)
target_link_libraries(asio_dispatch ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY})

add_executable(asio_strand_no asio_33_strand_no.cpp)
target_link_libraries(asio_strand_no ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY})

add_executable(asio_strand asio_34_strand.cpp)
target_link_libraries(asio_strand ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY})

add_executable(asio_61_timer asio_61_timeout.cpp)
target_link_libraries(asio_61_timer ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY})

add_executable(asio_62_timer asio_62_interval.cpp)
target_link_libraries(asio_62_timer ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY})

add_executable(asio_63 asio_63.cpp)
target_link_libraries(asio_63 ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY})

add_executable(asio_71_tcp_sync asio_71_tcp_sync.cpp)
target_link_libraries(asio_71_tcp_sync pthread ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY})

add_executable(asio_72_tcp_async asio_72_tcp_async.cpp)
target_link_libraries(asio_72_tcp_async pthread ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY})

add_executable(asio_81 asio_81.cpp)
target_link_libraries(asio_81 pthread ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY})


add_executable(bind_tutorial bind_tutorial.cpp)
target_link_libraries(bind_tutorial ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY})