cmake_minimum_required(VERSION 3.15)
project(coroutines)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++ -v")
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++ -lc++abi")

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_executable(main main.cpp)
target_compile_options(main PRIVATE -fcoroutines)

add_executable(echo_server echo_server.cpp)
target_compile_options(echo_server PRIVATE -fcoroutines)
target_compile_definitions(echo_server PUBLIC ASIO_HAS_CO_AWAIT)
target_compile_definitions(echo_server PUBLIC ASIO_HAS_STD_COROUTINE)
target_link_libraries(echo_server pthread)

add_executable(first_coroutine first_coroutine.cpp)
target_compile_options(first_coroutine PRIVATE -fcoroutines)

