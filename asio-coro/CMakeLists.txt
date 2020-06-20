cmake_minimum_required(VERSION 3.15)
project(coroutines)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++ -v")
#set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -stdlib=libc++ -lc++abi")

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

foreach(APP echo_server chat_server)
    add_executable(${APP} ${APP}.cpp)
    target_compile_options(${APP} PRIVATE -fcoroutines)
    target_compile_definitions(${APP} PUBLIC ASIO_HAS_CO_AWAIT)
    target_compile_definitions(${APP} PUBLIC ASIO_HAS_STD_COROUTINE)
    target_link_libraries(${APP} pthread)
endforeach()
