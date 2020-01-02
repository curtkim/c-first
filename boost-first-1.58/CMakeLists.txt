cmake_minimum_required(VERSION 2.8.12)
project(boost-first-1-58)

add_definitions("-std=c++11")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wformat=0")

find_package(Boost 1.58 REQUIRED system thread filesystem)

add_executable(thread_pool2 thread_pool2.cpp)
target_link_libraries(thread_pool2 ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY} ${Boost_REGEX_LIBRARY})