cmake_minimum_required(VERSION 2.8.12)
project(PackageTest CXX)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/cpx.json
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)


add_executable(example example.cpp)
target_link_libraries(example ${CONAN_LIBS})


# CTest is a testing tool that can be used to test your project.
# enable_testing()
# add_test(NAME example
#          WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
#          COMMAND example)
