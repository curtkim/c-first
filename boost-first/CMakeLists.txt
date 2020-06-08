cmake_minimum_required(VERSION 2.8.12)
project(boost-first)

add_definitions("-std=c++11")

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wformat=0")

add_executable(thread_pool thread_pool.cpp)
target_link_libraries(thread_pool ${CONAN_LIBS})

add_executable(thread_pool2 thread_pool2.cpp)
target_link_libraries(thread_pool2 ${CONAN_LIBS})

add_executable(thread_pool3 thread_pool3.cpp)
target_link_libraries(thread_pool3 ${CONAN_LIBS})

add_executable(program_options program_options.cpp)
target_link_libraries(program_options ${CONAN_LIBS})

add_executable(date_time date_time.cpp)
target_link_libraries(date_time ${CONAN_LIBS})

add_executable(test_fiber test_fiber.cpp)
target_link_libraries(test_fiber ${CONAN_LIBS})
