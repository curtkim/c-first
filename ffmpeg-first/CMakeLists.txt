cmake_minimum_required(VERSION 3.15)
project(video-nvidia-codec LANGUAGES C )

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_executable(encode_videon encode_videon.c)
target_link_libraries(encode_videon PUBLIC avcodec avutil) # avformat avfilter

add_executable(muxing muxing.c)
target_link_libraries(muxing PUBLIC m avcodec avformat avutil swscale swresample)