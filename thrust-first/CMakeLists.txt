cmake_minimum_required(VERSION 3.15)
project(thrust-first LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

foreach(APP 01_vector 02_vector 03_transform 04_tranform_saxpy 05_reduction 06_tiled_range)
    add_executable(${APP} ${APP}.cu)
    target_compile_features(${APP} PUBLIC cxx_std_17)
    target_include_directories(${APP} PUBLIC /usr/local/cuda/include)
    target_link_directories(${APP} PUBLIC /usr/local/cuda/lib64)
    target_link_libraries(${APP} PUBLIC ${CONAN_LIBS})
endforeach()