cmake_minimum_required(VERSION 3.16)
project(ring-span-my)

set(CMAKE_CXX_STANDARD 17)

add_executable(test ring-span-test.cpp)
target_compile_features(test PUBLIC cxx_std_17)

#foreach(APP
#        01-array 02-no-exceptions 03-make-ring-span 04-clear 05_pushback 06_reference 07_vector)
#    add_executable(${APP} ${APP}.cpp)
#    target_link_libraries(${APP} ${CONAN_LIBS})
#    target_compile_features(${APP} PUBLIC cxx_std_17)
#endforeach()
