cmake_minimum_required(VERSION 3.15)
project(coro-third-cpp)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

foreach(APP
        01_first
        02_echo_server
        03_http_client
        04_file_server
        05_link_cp
        06_timeout
        )
    add_executable(${APP} ${APP}.cpp)
    target_link_libraries(${APP} PUBLIC ${CONAN_LIBS} uring)
    target_compile_options(${APP} PRIVATE -fcoroutines)
    target_compile_features(${APP} PUBLIC cxx_std_20)
endforeach()

