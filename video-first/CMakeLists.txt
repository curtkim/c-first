cmake_minimum_required(VERSION 3.15)
project(video-first LANGUAGES CXX C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenGL REQUIRED)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()


include_directories(/home/curt/projects/c/Video_Codec_SDK_10.0.26/Interface)

foreach(APP pl_mpeg_extract_frames )
    add_executable(${APP} ${APP}.cpp)
    target_link_libraries(${APP} PUBLIC ${CONAN_LIBS})
    target_compile_features(${APP} PUBLIC cxx_std_17)
    target_include_directories(${APP} PRIVATE /usr/local/cuda/include)
endforeach()


add_executable(nv_extract_frames nv_extract_frames.cpp common/NvDecoder/NvDecoder.cpp)
target_link_directories(nv_extract_frames PUBLIC /home/curt/projects/c/Video_Codec_SDK_10.0.26/Lib/linux/stubs/x86_64)
target_link_directories(nv_extract_frames PUBLIC /usr/local/cuda/lib64)
target_link_libraries(nv_extract_frames PUBLIC ${CONAN_LIBS} nvcuvid nvidia-encode cuda cudart avcodec avformat avutil)
target_compile_features(nv_extract_frames PUBLIC cxx_std_17)
target_include_directories(nv_extract_frames PRIVATE /usr/local/cuda/include)

add_executable(simple_ffmpeg_info simple_ffmpeg_info.cpp)
target_link_libraries(simple_ffmpeg_info PUBLIC ${CONAN_LIBS} avcodec avformat avutil avfilter)
target_compile_features(simple_ffmpeg_info PUBLIC cxx_std_17)


add_executable(simple_ffmpeg_decode_pure simple_ffmpeg_decode_pure.cpp)
target_link_libraries(simple_ffmpeg_decode_pure PUBLIC ${CONAN_LIBS} avcodec avutil)
target_compile_features(simple_ffmpeg_decode_pure PUBLIC cxx_std_17)

file(COPY resources/bigbuckbunny_480x272.h264 DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)
file(COPY resources/bigbuckbunny_480x272.hevc DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)
file(COPY resources/bigbuckbunny_480x272.m2v DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)

add_executable(simple_ffmpeg_decode simple_ffmpeg_decode.cpp)
target_link_libraries(simple_ffmpeg_decode PUBLIC ${CONAN_LIBS} avcodec avformat avutil swscale)
target_compile_features(simple_ffmpeg_decode PUBLIC cxx_std_17)
file(COPY resources/Titanic.mkv DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)


add_executable(webcam_opengl webcam_opengl.c)
target_link_libraries(webcam_opengl PUBLIC ${CONAN_LIBS} ${OPENGL_LIBRARY} X11 v4l1 v4l2)
target_compile_features(webcam_opengl PUBLIC cxx_std_17)