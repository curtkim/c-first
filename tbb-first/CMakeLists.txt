cmake_minimum_required(VERSION 2.8.12)
project(MD5Encrypter)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

file(COPY input1.png DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)
file(COPY input2.png DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)

foreach(APP
        ex01_04 ex01_05 ex01_07 ex01_10_flow ex01_11_parallel_for ex01_12_unseq
        ex02_10_parallel_reduce ex02_30_parallel_pipeline
        ex03_03 ex03_05 ex03_06 ex03_10
        ex04_04 ex04_11
        ex18_03)
    add_executable(${APP} ${APP}.cpp)
    target_link_libraries(${APP} ${CONAN_LIBS})
    target_compile_features(${APP} PUBLIC cxx_std_17)
endforeach()