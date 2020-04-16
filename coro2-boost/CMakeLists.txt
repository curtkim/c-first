cmake_minimum_required(VERSION 3.15)
project(coro2)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_executable(fibo fibo.cpp)
target_link_libraries(fibo ${CONAN_LIBS})

add_executable(layout layout.cpp)
target_link_libraries(layout ${CONAN_LIBS})

add_executable(parser parser.cpp)
target_link_libraries(parser ${CONAN_LIBS})

add_executable(fringe fringe.cpp)
target_link_libraries(fringe ${CONAN_LIBS})

