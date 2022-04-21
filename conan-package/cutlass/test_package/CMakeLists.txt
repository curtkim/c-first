cmake_minimum_required(VERSION 3.20)
project(PackageTest CUDA CXX)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_executable(example example.cu)
target_link_libraries(example ${CONAN_LIBS})
target_compile_features(example PUBLIC cxx_std_17)

# CTest is a testing tool that can be used to test your project.
# enable_testing()
# add_test(NAME example
#          WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/bin
#          COMMAND example)
