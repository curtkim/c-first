cmake_minimum_required(VERSION 3.15)
project(boost-gil)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/test.jpg
        DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)

foreach(APP resize affine)
    add_executable(${APP} ${APP}.cpp)
    target_link_libraries(${APP} ${CONAN_LIBS})
    target_compile_features(${APP} PRIVATE cxx_std_17)
endforeach()
