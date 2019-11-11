 cmake_minimum_required(VERSION 2.8.12)
 project(conan-beast)

 add_definitions("-std=c++11")

 include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
 conan_basic_setup()

 add_executable(websocket-client main.cpp)
 target_link_libraries(websocket-client ${CONAN_LIBS})
