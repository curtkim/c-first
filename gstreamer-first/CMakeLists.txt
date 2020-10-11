cmake_minimum_required(VERSION 3.15)
project(gstreamer-first)

#include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
#conan_basic_setup()

include_directories(
        /usr/include/gstreamer-1.0
        /usr/include/glib-2.0
        /usr/lib/x86_64-linux-gnu/glib-2.0/include/)

foreach(APP basic-tutorial-1 basic-tutorial-2 basic-tutorial-3-dynamic-pipeline)
    add_executable(${APP} ${APP}.c)
    target_link_libraries(${APP} PUBLIC
            gstreamer-1.0
            gobject-2.0
            glib-2.0)
    #target_compile_features(basic-tutorial-1 PUBLIC cxx_std_20)
endforeach()

add_executable(main main.cpp)
target_link_libraries(main PUBLIC
        gstreamer-1.0
        gobject-2.0
        glib-2.0
        )
