cmake_minimum_required(VERSION 3.15)
project(functional)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

foreach(APP 10_monad_vector 10_functors-optional 10_monad-optional 10_monad_range)
    add_executable(${APP} ${APP}.cpp)
    target_compile_features(${APP} PUBLIC cxx_std_17)
endforeach()
