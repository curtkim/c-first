cmake_minimum_required(VERSION 2.8.12)
project(boost-first)

add_definitions("-std=c++17")

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wformat=0")

foreach(APP
        test-coroutine
        test-coroutine2
        #test-coroutine-echo-server
        test-spawn-echo-server
        )
    add_executable(${APP} ${APP}.cpp)
    target_link_libraries(${APP} ${CONAN_LIBS})
endforeach()