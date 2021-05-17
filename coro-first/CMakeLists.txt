cmake_minimum_required(VERSION 3.15)
project(coro_first)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++ -v")
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++ -lc++abi")

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

foreach(APP
        00_helloworld
        01_generator 01_nested
        02_my_first_coroutine 02_my_first_coroutine_return 02_my_first_coroutine_yield
        04_state_machine 05_task
        06_future
        11_russian-roulette
        )
    add_executable(${APP} ${APP}.cpp)
    target_compile_options(${APP} PRIVATE -fcoroutines)
    target_compile_features(${APP} PUBLIC cxx_std_20)
endforeach()
target_link_libraries(06_future pthread)

foreach(APP 21_linux_event_wait )
    add_executable(${APP} ${APP}.cpp coroutine/linux.cpp )
    #target_link_libraries(21_linux_event_wait PUBLIC coroutine_linux)
    target_include_directories(${APP} PUBLIC .)
    target_compile_definitions(${APP} PUBLIC gsl_CONFIG_DEFAULTS_VERSION)
    target_compile_options(${APP} PRIVATE -fcoroutines)
    target_compile_features(${APP} PUBLIC cxx_std_20)
endforeach()


add_executable(22_net_socket_tcp_echo 22_net_socket_tcp_echo.cpp coroutine/io_linux.cpp coroutine/linux.cpp latch_linux.cpp socket.cpp)
#target_link_libraries(21_linux_event_wait PUBLIC coroutine_linux)
target_include_directories(22_net_socket_tcp_echo PUBLIC .)
target_compile_definitions(22_net_socket_tcp_echo PUBLIC gsl_CONFIG_DEFAULTS_VERSION)
target_compile_options(22_net_socket_tcp_echo PRIVATE -fcoroutines)
target_compile_features(22_net_socket_tcp_echo PUBLIC cxx_std_20)


add_executable(03_echo_server 03_echo_server.cpp)
target_compile_options(03_echo_server PRIVATE -fcoroutines)
target_compile_definitions(03_echo_server PUBLIC ASIO_HAS_CO_AWAIT)
target_compile_definitions(03_echo_server PUBLIC ASIO_HAS_STD_COROUTINE)
target_link_libraries(03_echo_server pthread)

add_executable(31_simcpp 31_simcpp_main.cpp 31_simcpp.cpp)
target_compile_options(31_simcpp PRIVATE -fcoroutines)