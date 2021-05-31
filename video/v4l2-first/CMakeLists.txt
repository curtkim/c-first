cmake_minimum_required(VERSION 3.15)
project(v4l2-first)

set(VIDEO_SDK_VERSION 11.0.10)


include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()


add_executable(01_vidtest 01_vidtest.cpp)
target_link_libraries(01_vidtest PUBLIC ${CONAN_LIBS})

add_library(capture_utils capture_utils.cpp)


foreach(APP
        my_capture_mmap_epoll
        my_capture_mmap_epoll_multi
        my_capture_mmap_uring_poll)
    add_executable(${APP} ${APP}.cpp)
    target_link_libraries(${APP} PUBLIC ${CONAN_LIBS} capture_utils uring)

    target_include_directories(${APP} PRIVATE /usr/local/cuda/include)
    target_include_directories(${APP} PRIVATE /opt/nvidia/Video_Codec_SDK_${VIDEO_SDK_VERSION}/Interface)
    target_link_directories(${APP} PUBLIC /opt/nvidia/Video_Codec_SDK_${VIDEO_SDK_VERSION}/Lib/linux/stubs/x86_64)
    target_link_directories(${APP} PUBLIC /usr/local/cuda/lib64)
    target_link_libraries(${APP} PUBLIC nvcuvid nvidia-encode cuda cudart
            nvcuvid nvidia-encode cuda cudart nppicc nvToolsExt)
endforeach()


foreach(APP
        my_capture_all
        my_capture_mmap
        my_capture_mmap_poll)
    add_executable(_temp_${APP} temp/${APP}.cpp)
    target_link_libraries(_temp_${APP} PUBLIC ${CONAN_LIBS} )
endforeach()

