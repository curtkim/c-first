cmake_minimum_required(VERSION 3.18)
project(klein-first)

#set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

message("${CONAN_LIBS}")

foreach(APP
        01_quick_start 02_angle_between_plane 03_line_from_plan_intersection 04_join
        05_blend_motors
        )
    add_executable(${APP} ${APP}.cpp)
    target_link_libraries(${APP} PUBLIC ${CONAN_LIBS} )
    target_compile_features(${APP} PUBLIC cxx_std_17)

    target_compile_options(${APP} PUBLIC -msse3) #klein cmake에서 처리되면 좋은데..
endforeach()