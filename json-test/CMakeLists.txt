cmake_minimum_required(VERSION 2.8.12)
project(json_test)

#set(CMAKE_CXX_STANDARD 17)

if(EXISTS ${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
    include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
    conan_basic_setup()
else()
    message(WARNING "The file conanbuildinfo.cmake doesn't exist, you have to run conan install first")
endif()

add_definitions("-std=c++17")

add_executable(json_test test.cpp)
target_link_libraries(json_test ${CONAN_LIBS})

# copy test fixture
file(COPY file.json DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)