cmake_minimum_required(VERSION 3.15)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(thrust-first LANGUAGES CXX CUDA)

#set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

foreach(APP 00_struct
        01_vector 02_vector 03_transform 04_tranform_saxpy 05_reduction 06_tiled_range
        07_reduce_by_key 08_bouding_box 09_bucket_sort2d
        10_device_ptr
        11_iterator

        51_stream_concurrent
        91_with_eigen)
    add_executable(${APP} ${APP}.cu )
    target_compile_features(${APP} PUBLIC cxx_std_17)
    target_include_directories(${APP} PUBLIC /usr/local/cuda/include)
    target_link_directories(${APP} PUBLIC /usr/local/cuda/lib64)
    target_link_libraries(${APP} PUBLIC ${CONAN_LIBS})
    set_target_properties(${APP} PROPERTIES CUDA_ARCHITECTURES 61)
endforeach()