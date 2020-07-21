cmake_minimum_required(VERSION 3.15)
project(gsl)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_executable(main main.cpp)
target_link_libraries(main ${CONAN_LIBS})
target_compile_features(main PUBLIC cxx_std_17)
