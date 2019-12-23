cmake_minimum_required(VERSION 2.8.12)
project(json_test)

set(CMAKE_CXX_STANDARD 17)

if(EXISTS ${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
    include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
    conan_basic_setup()
else()
    message(WARNING "The file conanbuildinfo.cmake doesn't exist, you have to run conan install first")
endif()

#add_definitions("-std=c++17")

add_executable(test test.cpp)
target_link_libraries(test ${CONAN_LIBS})

add_executable(test2 test2.cpp)
target_link_libraries(test2 ${CONAN_LIBS})

add_executable(test_pplx test_pplx.cpp)
target_link_libraries(test_pplx ${CONAN_LIBS})
