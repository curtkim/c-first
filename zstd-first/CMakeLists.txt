cmake_minimum_required(VERSION 3.15)
project(zstd_first CXX)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup(TARGETS)

foreach(APP
        main)
    add_executable(${APP} ${APP}.cpp)
    target_link_libraries(${APP} ${CONAN_LIBS})
    target_compile_features(${APP} PUBLIC cxx_std_17)
endforeach()
