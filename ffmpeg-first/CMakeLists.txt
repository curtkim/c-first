cmake_minimum_required(VERSION 3.15)
project(video-nvidia-codec LANGUAGES C CXX )

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_executable(encode_video encode_video.c)
target_link_libraries(encode_video PUBLIC avcodec avutil) # avformat avfilter

add_executable(encode_audio encode_audio.c)
target_link_libraries(encode_audio PUBLIC m avcodec avutil) # avformat avfilter

add_executable(muxing muxing.c)
target_link_libraries(muxing PUBLIC m avcodec avformat avutil swscale swresample)

add_executable(decode_video decode_video.c)
target_link_libraries(decode_video PUBLIC avcodec avutil) # avformat avfilter

add_executable(decode_video_cpp decode_video.cpp)
target_link_libraries(decode_video_cpp PUBLIC avcodec avutil) # avformat avfilter


add_executable(demuxing demuxing.c)
target_link_libraries(demuxing PUBLIC m avcodec avformat avutil swscale swresample)
target_compile_features(demuxing PUBLIC cxx_std_17)