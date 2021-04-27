cmake_minimum_required(VERSION 3.18)
project(cairo-first)

#set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

foreach(APP
        01_babel
        02_arc

        )
    add_executable(${APP} ${APP}.c)
    target_link_libraries(${APP} PUBLIC ${CONAN_LIBS} )
    target_compile_features(${APP} PUBLIC cxx_std_17)
endforeach()