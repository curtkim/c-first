cmake_minimum_required(VERSION 3.15)
project(catch2)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()


## single file
add_executable(10 10-TestCase.cpp)
target_link_libraries(010 ${CONAN_LIBS})

## normal(assert / fixture)
add_executable(30 30-Main.cpp
        31-Assert-Require-Check.cpp
        32_Assert_Require_that.cpp
        35-Fixture-Section.cpp
        36-Fixture-ClassFixture.cpp
        )
target_link_libraries(030 ${CONAN_LIBS})
target_compile_features(030 PRIVATE cxx_std_17)

## generator
add_executable(50 50-Main.cpp
        51-Generator.cpp
        52-Generator-MapTypeConversion.cpp
        53-Generator-Variables.cpp
        )
target_link_libraries(050 ${CONAN_LIBS})
target_compile_features(050 PRIVATE cxx_std_17)