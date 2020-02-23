cmake_minimum_required(VERSION 2.8.12)
project(boost-first-1-58)

add_definitions("-std=c++11")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wformat=0")

find_package(Boost 1.65 REQUIRED system thread filesystem)

foreach(APP
        thread_pool2
        asio_01_single_thread_run_and_poll
        asio_02_multi_thread_global
        asio_03_multi_thread
        asio_04_multi_thread_tutorial_lock
        asio_31_post
        asio_32_dispatch
        asio_33_strand_no
        asio_34_strand
        asio_61_timeout
        asio_62_interval
        asio_63
        asio_71_tcp_sync
        asio_72_tcp_async
        asio_81
        bind_tutorial
        tcp_async_echo_server
        ex32_1
        ex32_2
        ex32_3
        ex32_5
        ex32_6
        ex32_10
        )
    add_executable(${APP} ${APP}.cpp)
    target_link_libraries(${APP} pthread ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY})
endforeach()

add_executable(chat_client chat_client.cpp chat_message.hpp)
target_link_libraries(chat_client pthread ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY})

add_executable(chat_server chat_server.cpp chat_message.hpp)
target_link_libraries(chat_server pthread ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY})


get_cmake_property(_variableNames VARIABLES)
list (SORT _variableNames)
foreach (_variableName ${_variableNames})
    message(STATUS "${_variableName}=${${_variableName}}")
endforeach()


add_executable(serialization_client serialization/client.cpp serialization/stock.hpp serialization/connection.hpp)
target_link_libraries(serialization_client pthread ${Boost_LIBRARY_DIRS}/libboost_serialization.so ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY})

add_executable(serialization_server serialization/server.cpp serialization/stock.hpp serialization/connection.hpp)
target_link_libraries(serialization_server pthread ${Boost_LIBRARY_DIRS}/libboost_serialization.so ${Boost_SYSTEM_LIBRARY} ${Boost_THREAD_LIBRARY})
