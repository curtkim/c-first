cmake_minimum_required(VERSION 3.15)
project(test_package C)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()


foreach(A ${CONAN_LIBS})
  message(STATUS ${A})
endforeach()

#find_package(zstd REQUIRED)

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/logo.png
    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)

add_executable(${PROJECT_NAME} test.c)
target_link_libraries(${PROJECT_NAME} ${CONAN_LIBS})

