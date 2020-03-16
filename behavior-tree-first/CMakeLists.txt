cmake_minimum_required(VERSION 2.8.12)
project(MD5Encrypter)

add_definitions("-std=c++11")

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_library(dummy_nodes dummy_nodes.cpp )
target_link_libraries(dummy_nodes ${CONAN_LIBS})

add_library(movebase_node movebase_node.cpp )
target_link_libraries(movebase_node ${CONAN_LIBS})

add_library(closedoor_nodes closedoor_nodes.cpp )
target_link_libraries(closedoor_nodes ${CONAN_LIBS})


foreach(APP tutorial01 tutorial02 tutorial05_close_door)
    message(${APP})
    add_executable(${APP} ${APP}.cpp)
    target_link_libraries(${APP} ${CONAN_LIBS} dummy_nodes movebase_node closedoor_nodes dl) # dl은 so 관련 함수가 있는 system library인 것 같다.
endforeach()

