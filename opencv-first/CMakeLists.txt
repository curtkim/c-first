cmake_minimum_required(VERSION 2.8.12)
project(MD5Encrypter)

add_definitions("-std=c++17")

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_executable(lena lena.cpp)
target_link_libraries(lena ${CONAN_LIBS})

add_executable(sobel sobel.cpp)
target_link_libraries(sobel ${CONAN_LIBS})
