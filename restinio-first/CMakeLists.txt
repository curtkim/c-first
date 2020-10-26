cmake_minimum_required(VERSION 3.17)
project(restinio)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

foreach(A ${CONAN_LIBS})
    message(STATUS ${A})
endforeach()

add_executable(main main.cpp)
target_link_libraries(main PUBLIC ${CONAN_LIBS} )
target_compile_features(main PRIVATE cxx_std_17)