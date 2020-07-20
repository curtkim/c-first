cmake_minimum_required(VERSION 3.15)
project(libjpeg)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()


file(COPY sample.jpg DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)

add_executable(encode encode.cpp)
target_link_libraries(encode ${CONAN_LIBS_LIBJPEG-TURBO})

add_executable(JpegLoaderMain JpegLoader.cpp JpegLoaderMain.cpp)
target_link_libraries(JpegLoaderMain ${CONAN_LIBS_LIBJPEG-TURBO})
