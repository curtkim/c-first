cmake_minimum_required(VERSION 2.8)
project(coroutines)

#set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Werror")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++2a")

#set(SOURCE_FILES main.cpp)

foreach(APP fibonacci_generator)
    add_executable(${APP} ${APP}.cpp)
    target_compile_options(${APP} PRIVATE -fcoroutines)
endforeach()

#target_link_libraries(${PROJECT_NAME} rt dl pthread)
