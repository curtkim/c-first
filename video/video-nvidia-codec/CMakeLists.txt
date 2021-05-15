cmake_minimum_required(VERSION 3.15)
project(video-nvidia-codec LANGUAGES CXX C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(VIDEO_SDK_VERSION 11.0.10)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

# 2 nvidia env/dec
foreach(APP
        01_encode)
    add_executable(${APP} ${APP}.cpp)
    target_include_directories(${APP} PRIVATE /usr/local/cuda/include)
    target_include_directories(${APP} PRIVATE /opt/nvidia/Video_Codec_SDK_${VIDEO_SDK_VERSION}/Interface)
    target_link_directories(${APP} PUBLIC /opt/nvidia/Video_Codec_SDK_${VIDEO_SDK_VERSION}/Lib/linux/stubs/x86_64)
    target_link_directories(${APP} PUBLIC /usr/local/cuda/lib64)
    target_link_libraries(${APP} PUBLIC ${CONAN_LIBS} nvcuvid nvidia-encode cuda cudart)
    target_compile_features(${APP} PUBLIC cxx_std_17)
endforeach()

#target_link_libraries(04_encode_egl PUBLIC EGL ${OPENGL_LIBRARY})

#add_executable(yuv2rgb yuv2rgb)
#target_link_libraries(yuv2rgb PUBLIC ${CONAN_LIBS})
#target_compile_features(yuv2rgb PUBLIC cxx_std_17)
