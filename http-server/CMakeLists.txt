cmake_minimum_required (VERSION 3.2 FATAL_ERROR)
project (http-server)

set (CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Boost 1.70.0 REQUIRED COMPONENTS system thread) #  regex asio
include_directories("${Boost_INCLUDE_DIRS}")

# add the executable
add_executable(http-server main.cpp fields_alloc.hpp client.cpp)
set_property(TARGET http-server PROPERTY CXX_STANDARD 17)
target_link_libraries (http-server ${Boost_LIBRARIES})

add_executable(http-client client.cpp )
set_property(TARGET http-client PROPERTY CXX_STANDARD 17)
target_link_libraries (http-client ${Boost_LIBRARIES})
