cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)

project(ChuckNorris)

add_library(chucknorris
    include/ChuckNorris.hpp
    src/ChuckNorris.cpp
)


target_include_directories(
  chucknorris
  PUBLIC
    "include"
)

add_executable(cpp_demo
  src/main.cpp
)

target_link_libraries(cpp_demo chucknorris)