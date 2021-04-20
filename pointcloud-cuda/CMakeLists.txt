cmake_minimum_required(VERSION 3.18)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(thrust-first LANGUAGES CXX CUDA)

#set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

foreach(APP 01_read_ply)
    add_executable(${APP} ${APP}.cu )
    target_compile_features(${APP} PUBLIC cxx_std_17)
    target_include_directories(${APP} PUBLIC /usr/local/cuda/include)
    target_link_directories(${APP} PUBLIC /usr/local/cuda/lib64)
    target_link_libraries(${APP} PUBLIC ${CONAN_LIBS})
    set_target_properties(${APP} PROPERTIES CUDA_ARCHITECTURES 61)
endforeach()

foreach(APP 11_tinyply_test 12_pointcloud_test 13_pointcloud_minmax)
    add_executable(${APP} ${APP}.cpp )
    target_include_directories(${APP} PUBLIC .)
    target_compile_features(${APP} PUBLIC cxx_std_17)
    target_link_libraries(${APP} PUBLIC ${CONAN_LIBS})
endforeach()