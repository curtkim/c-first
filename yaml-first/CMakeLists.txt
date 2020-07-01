cmake_minimum_required(VERSION 2.8.12)
project(MD5Encrypter)

add_definitions("-std=c++11")

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_executable(main main.cpp)
target_link_libraries(main ${CONAN_LIBS})

add_executable(convert convert.cpp)
target_link_libraries(convert ${CONAN_LIBS})

add_executable(nested nested.cpp)
target_link_libraries(nested ${CONAN_LIBS})
