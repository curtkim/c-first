cmake_minimum_required(VERSION 3.15)
project(ut-first)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

foreach(APP 01_first 02_nest 03_spec_describe_it 04_parameterized 05_suite 06_matcher 07_throw)
    add_executable(${APP} ${APP}.cpp)
    target_compile_features(${APP} PUBLIC cxx_std_20)
    target_link_libraries(${APP} ${CONAN_LIBS})
endforeach()

#add_definitions("-std=c++17")

#add_executable(test-runner test-runner.cpp ${TEST_SOURCES})
