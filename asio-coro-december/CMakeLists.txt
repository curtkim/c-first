cmake_minimum_required(VERSION 3.15)
project(asio-coro-december)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++ -v")
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++ -lc++abi")

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()


add_executable(main src/main.cpp
        src/async/condition_variable_impl.cpp
        src/async/stop_source.cpp
        src/websocket/connect_options.cpp
        src/websocket/connection.cpp
        src/websocket/connection_impl.cpp
        src/websocket/event.cpp
        src/websocket/message.cpp
        src/websocket/variant_websocket.cpp)
target_compile_options(main PRIVATE -fcoroutines)
target_compile_definitions(main PUBLIC ASIO_HAS_CO_AWAIT)
target_compile_definitions(main PUBLIC ASIO_HAS_STD_COROUTINE)
target_link_libraries(main ${CONAN_LIBS} pthread)
target_include_directories(main PUBLIC src)
target_compile_features(main PUBLIC cxx_std_20)

add_executable(function-try-blocks function-try-blocks.cpp)