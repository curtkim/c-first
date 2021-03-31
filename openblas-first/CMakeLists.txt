cmake_minimum_required(VERSION 3.15)
project(openblas-first LANGUAGES CXX)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

foreach(APP cblas_dgemm cblas_dgemm2)
    add_executable(${APP} ${APP}.cpp)
    target_compile_features(${APP} PRIVATE cxx_std_17)
    target_link_libraries(${APP} ${CONAN_LIBS})
endforeach()