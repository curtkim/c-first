cmake_minimum_required(VERSION 3.15)
project(v4l2-first)

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
endforeach()


foreach(APP
        my_capture_all
        my_capture_mmap
        my_capture_mmap_poll)
    add_executable(_temp_${APP} temp/${APP}.cpp)
    target_link_libraries(_temp_${APP} PUBLIC ${CONAN_LIBS} )
endforeach()

