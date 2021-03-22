cmake_minimum_required(VERSION 3.15)
project(conveyor_belt/0.1.0)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

foreach(APP
        01-basic )
    add_executable(${APP} ${APP}.cpp)
    target_link_libraries(${APP} ${CONAN_LIBS})
    target_compile_features(${APP} PUBLIC cxx_std_17)
endforeach()


set(TEST_SOURCES
        track-test.cpp
        )

add_executable(test test.cpp ${TEST_SOURCES})