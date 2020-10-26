cmake_minimum_required(VERSION 3.15)
project(video-first LANGUAGES CXX C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

find_package(OpenGL REQUIRED)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()


include_directories(/home/curt/projects/c/Video_Codec_SDK_10.0.26/Interface)

# 1
foreach(APP pl_mpeg_extract_frames )
    add_executable(${APP} ${APP}.cpp)
    target_link_libraries(${APP} PUBLIC ${CONAN_LIBS})
    target_compile_features(${APP} PUBLIC cxx_std_17)
    target_include_directories(${APP} PRIVATE /usr/local/cuda/include)
endforeach()

# 2 nvidia env/dec
add_executable(nv_extract_frames nv_extract_frames.cpp common/NvDecoder/NvDecoder.cpp)
target_link_directories(nv_extract_frames PUBLIC /home/curt/projects/c/Video_Codec_SDK_10.0.26/Lib/linux/stubs/x86_64)
target_link_directories(nv_extract_frames PUBLIC /usr/local/cuda/lib64)
target_link_libraries(nv_extract_frames PUBLIC ${CONAN_LIBS} nvcuvid nvidia-encode cuda cudart avcodec avformat avutil)
target_compile_features(nv_extract_frames PUBLIC cxx_std_17)
target_include_directories(nv_extract_frames PRIVATE /usr/local/cuda/include)

# 3. ffmpeg
foreach(APP simple_ffmpeg_info simple_ffmpeg_decode_pure simple_ffmpeg_decode)
    add_executable(${APP} ${APP}.cpp)
    target_link_libraries(${APP} PUBLIC ${CONAN_LIBS} avcodec avformat avutil avfilter swscale)
    target_compile_features(${APP} PUBLIC cxx_std_17)
endforeach()

#foreach(APP simple_ffmpeg_info )
#    add_executable(${APP}_cpp ${APP}.cpp)
#    target_link_libraries(${APP}_cpp PUBLIC ${CONAN_LIBS} avcodec avformat avutil avfilter)
#    target_compile_features(${APP}_cpp PUBLIC cxx_std_17)
#endforeach()


file(COPY resources/bigbuckbunny_480x272.h264 DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)
file(COPY resources/bigbuckbunny_480x272.hevc DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)
file(COPY resources/bigbuckbunny_480x272.m2v DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)
file(COPY resources/Titanic.mkv DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)

# 4. webcam_opengl
add_executable(webcam_opengl webcam_opengl.c)
target_link_libraries(webcam_opengl PUBLIC ${CONAN_LIBS} ${OPENGL_LIBRARY} X11 v4l1 v4l2)
target_compile_features(webcam_opengl PUBLIC cxx_std_17)

add_executable(webcam_opengl2 webcam_opengl2.cpp)
target_link_libraries(webcam_opengl2 PUBLIC ${CONAN_LIBS} ${OPENGL_LIBRARY} X11 v4l1 v4l2)
target_compile_features(webcam_opengl2 PUBLIC cxx_std_17)


# 5. ffmpeg - opegnl
foreach(APP ffmpeg_extract_ppm)
    add_executable(${APP} ${APP}.cpp)
    target_link_libraries(${APP} PUBLIC ${CONAN_LIBS} avcodec avformat avutil avfilter swscale) # ${OPENGL_LIBRARY} X11
    target_compile_features(${APP} PUBLIC cxx_std_17)
endforeach()
