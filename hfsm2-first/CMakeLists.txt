cmake_minimum_required(VERSION 3.15)
project(hfsm2-first)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

foreach(APP 01_tutorial example)
    add_executable(${APP} ${APP}.cpp)
    target_link_libraries(${APP} ${CONAN_LIBS})
    target_compile_features(${APP} PRIVATE cxx_std_17)
endforeach()
