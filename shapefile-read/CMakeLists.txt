cmake_minimum_required(VERSION 2.8.12)
project(shapefile-read)

if(EXISTS ${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
    include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
    conan_basic_setup()
else()
    message(WARNING "The file conanbuildinfo.cmake doesn't exist, you have to run conan install first")
endif()

add_definitions("-std=c++11")

add_executable(main main.cpp)
#target_link_libraries(main)

add_executable(main2 main2.cpp)
target_link_libraries(main2 ${CONAN_LIBS})