cmake_minimum_required(VERSION 3.16)
project(conveyor_belt/0.1.0)

#set(CMAKE_VERBOSE_MAKEFILE ON)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

foreach(APP
        01-basic 11_carla)
    add_executable(${APP} ${APP}.cpp)

    target_precompile_headers(${APP} PRIVATE all.hpp)

    target_include_directories(${APP} PUBLIC /usr/local/cuda/include)
    target_link_directories(${APP} PUBLIC /usr/local/cuda/lib64)
    target_link_libraries(${APP} ${CONAN_LIBS} atomic nvToolsExt cuda cudart)
    target_compile_features(${APP} PUBLIC cxx_std_20)
endforeach()


set(TEST_SOURCES
        temp/struct-vector-test.cpp
        test/track-test.cpp
        test/timeline-test.cpp
        )

add_executable(test test/main.cpp ${TEST_SOURCES})
target_compile_features(test PUBLIC cxx_std_20)
target_link_libraries(test ${CONAN_LIBS})

## bench
add_executable(struct-vector-bench temp/struct-vector-bench.cpp)
target_link_libraries(struct-vector-bench ${CONAN_LIBS})