cmake_minimum_required(VERSION 3.17)
project(unifex-first)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

#foreach(A ${CONAN_LIBS})
#    message(STATUS ${A})
#endforeach()

add_executable(when_all when_all.cpp)
target_link_libraries(when_all PUBLIC ${CONAN_LIBS} )
target_compile_features(when_all PUBLIC cxx_std_17)