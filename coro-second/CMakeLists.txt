cmake_minimum_required(VERSION 3.15)
project(coro_seconds)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++ -v")
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++ -lc++abi")

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

foreach(APP 01_main 01_nested 04_state_machine 05_task
        02_my_first_coroutine 02_my_first_coroutine_return 02_my_first_coroutine_yield
        11_russian-roulette)
    add_executable(${APP} ${APP}.cpp)
    target_compile_options(${APP} PRIVATE -fcoroutines)
    target_compile_features(${APP} PUBLIC cxx_std_20)
endforeach()

add_executable(03_echo_server 03_echo_server.cpp)
target_compile_options(03_echo_server PRIVATE -fcoroutines)
target_compile_definitions(03_echo_server PUBLIC ASIO_HAS_CO_AWAIT)
target_compile_definitions(03_echo_server PUBLIC ASIO_HAS_STD_COROUTINE)
target_link_libraries(03_echo_server pthread)
