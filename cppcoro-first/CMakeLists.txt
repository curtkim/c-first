cmake_minimum_required(VERSION 3.15)
project(cppcoro-first)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++2a")

#set(SOURCE_FILES main.cpp)

foreach(APP 00_fibonacci_generator 00_chain
        01_sync_wait 02_static_thread_pool 03_static_thread_pool_sum 04_when_all
        11_task
        21_producer_consumer
        22_cancellation 23_mutex
        31_whenAllOnThreadPool 32_async_latch)
    add_executable(${APP} ${APP}.cpp)
    target_compile_options(${APP} PUBLIC -fcoroutines)
    target_link_libraries(${APP} PUBLIC ${CONAN_LIBS} pthread)
    target_compile_features(${APP} PUBLIC cxx_std_20)
    target_compile_definitions(${APP} PUBLIC CPPCORO_USE_IO_RING)
endforeach()

#target_link_libraries(${PROJECT_NAME} rt dl pthread)
