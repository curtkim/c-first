cmake_minimum_required(VERSION 2.8.12)
project(boost-first-1-58)

add_definitions("-std=c++11")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wformat=0")

find_package(Boost 1.58 REQUIRED system thread filesystem)

add_executable(thread_pool2 thread_pool2.cpp)
target_link_libraries(thread_pool2 ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY})

add_executable(asio_tutorial asio_tutorial.cpp)
target_link_libraries(asio_tutorial ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY})

add_executable(asio_multi_thread_tutorial asio_multi_thread_tutorial.cpp)
target_link_libraries(asio_multi_thread_tutorial ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY})

add_executable(asio_multi_thread_tutorial2 asio_multi_thread_tutorial2.cpp)
target_link_libraries(asio_multi_thread_tutorial2 ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY})

add_executable(asio_multi_thread_tutorial_lock asio_multi_thread_tutorial_lock.cpp)
target_link_libraries(asio_multi_thread_tutorial_lock ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY})

add_executable(asio_post asio_post.cpp)
target_link_libraries(asio_post ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY})

add_executable(asio_dispatch asio_dispatch.cpp)
target_link_libraries(asio_dispatch ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY})

add_executable(asio_strand_no asio_strand_no.cpp)
target_link_libraries(asio_strand_no ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY})

add_executable(asio_strand asio_strand.cpp)
target_link_libraries(asio_strand ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY})

add_executable(bind_tutorial bind_tutorial.cpp)
target_link_libraries(bind_tutorial ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY})