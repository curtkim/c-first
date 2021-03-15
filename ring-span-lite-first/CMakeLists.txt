cmake_minimum_required(VERSION 3.15)
project(ring-span-lite/0.4.0)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()


foreach(APP
        01-basic 02-no-exceptions 03-make-ring-span 04-clear 05_pushback 06_reference)
    add_executable(${APP} ${APP}.cpp)
    target_link_libraries(${APP} ${CONAN_LIBS})
    target_compile_features(${APP} PUBLIC cxx_std_17)
endforeach()
