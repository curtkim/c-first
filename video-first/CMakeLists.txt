cmake_minimum_required(VERSION 3.15)
project(range-first)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_executable(pl_mpeg_extract_frames pl_mpeg_extract_frames.cpp)
target_link_libraries(pl_mpeg_extract_frames PUBLIC ${CONAN_LIBS}) # header only library
target_compile_features(pl_mpeg_extract_frames PUBLIC cxx_std_17)
