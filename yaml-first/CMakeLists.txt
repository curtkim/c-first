cmake_minimum_required(VERSION 3.15)
project(yaml-first)

add_definitions("-std=c++11")

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_executable(main main.cpp)
target_link_libraries(main ${CONAN_LIBS})

add_executable(convert convert.cpp)
target_link_libraries(convert ${CONAN_LIBS})

add_executable(nested nested.cpp)
target_link_libraries(nested ${CONAN_LIBS})

add_executable(hydra hydra.cpp)
target_link_libraries(hydra ${CONAN_LIBS})
target_compile_features(hydra PUBLIC cxx_std_17)
