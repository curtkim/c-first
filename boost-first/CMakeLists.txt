cmake_minimum_required(VERSION 2.8.12)
project(boost-first)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wformat=0")

foreach(APP
        thread_pool thread_pool2 thread_pool3
        program_options
        date_time
        fiber_first fiber_ping_pong fiber_range_for fiber_adapt_nonblocking fiber_work_sharing)
    add_executable(${APP} ${APP}.cpp)
    target_link_libraries(${APP} ${CONAN_LIBS})
    target_compile_features(${APP} PUBLIC cxx_std_17)
endforeach()
