cmake_minimum_required(VERSION 3.16)
project(lsf-first LANGUAGES CXX)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

foreach(APP lsf)
    add_executable(${APP} ${APP}.cpp)
    target_compile_features(${APP} PRIVATE cxx_std_17)
    target_link_libraries(${APP} ${CONAN_LIBS})
endforeach()