cmake_minimum_required(VERSION 2.8.12)
project(conan-gtest)

if(EXISTS ${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
    include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
    conan_basic_setup()
else()
    message(WARNING "The file conanbuildinfo.cmake doesn't exist, you have to run conan install first")
endif()

add_definitions("-std=c++11")

include_directories(mymath)
add_subdirectory(mymath)

add_executable(main main.cpp)
target_link_libraries(main mymath)
