cmake_minimum_required(VERSION 3.15)
project(magic_enum-first LANGUAGES CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

foreach(APP example)
    add_executable(${APP} ${APP}.cpp)
    target_compile_features(${APP} PUBLIC cxx_std_17)
    target_link_libraries(${APP} PUBLIC ${CONAN_LIBS})
endforeach()