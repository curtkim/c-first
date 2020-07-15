cmake_minimum_required(VERSION 2.8.12)
project(MD5Encrypter)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_executable(ex3_03 ex3_03.cpp)
target_link_libraries(ex3_03 ${CONAN_LIBS})
target_compile_features(ex3_03 PUBLIC cxx_std_17)