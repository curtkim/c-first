cmake_minimum_required(VERSION 2.8.12)
project(spdlog-first)

set ( CMAKE_EXPORT_COMPILE_COMMANDS on )

add_definitions("-std=c++11")

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

foreach(APP first logger multi_sink user_defined_type)
    add_executable(${APP} ${APP}.cpp)
    target_link_libraries(${APP} ${CONAN_LIBS})
endforeach()
