cmake_minimum_required(VERSION 3.15)
project(v4l2-first)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_executable(01_vidtest 01_vidtest.cpp)
target_link_libraries(01_vidtest PUBLIC ${CONAN_LIBS})

add_library(capture_utils capture_utils.cpp)

foreach(APP
        my_capture_mmap_epoll)
    add_executable(${APP} ${APP}.cpp)
    target_link_libraries(${APP} PUBLIC ${CONAN_LIBS} capture_utils)
endforeach()



#add_library(capture OBJECT capture.cpp)
foreach(APP
        my_capture_all
        my_capture_mmap
        my_capture_mmap_poll
        my_capture_mmap_epoll2
        my_capture_mmap_epoll2_multi
        my_capture_mmap_uring_poll)
    add_executable(${APP} temp/${APP}.cpp)
    target_link_libraries(${APP} PUBLIC ${CONAN_LIBS} capture_utils)
endforeach()

target_link_libraries(my_capture_mmap_uring_poll PUBLIC uring)
