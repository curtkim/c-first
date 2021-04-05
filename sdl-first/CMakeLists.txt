cmake_minimum_required(VERSION 3.16)
project(sdl-first)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

find_package(SDL2 REQUIRED)
find_package(SDL2_image REQUIRED)
include_directories(.)

file(COPY "${PROJECT_SOURCE_DIR}/image.png" DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_executable(exam1 example1.cpp)
target_link_libraries(exam1 ${SDL2_LIBRARY})

add_executable(exam5 example5.cpp)
target_link_libraries(exam5 ${SDL2_LIBRARY} ${SDL2_IMAGE_LIBRARY})
