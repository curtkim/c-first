cmake_minimum_required(VERSION 3.22)
project(cuda_learn CUDA CXX)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

set(CMAKE_CUDA_STANDARD 17)
#set(CMAKE_CUDA_FLAGS "-Wall")
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

add_compile_options(-G) # to add CUDA debug symbols
add_compile_options(--generate-line-info)

foreach(APP 01_print_from_gpu 12_vector_add )
    add_executable(${APP} ${APP}.cu)
    #target_compile_options(
    #        01_print_from_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
    #        --generate-line-info>)
    set_target_properties(${APP} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endforeach()

foreach(APP 81_cublas_sgemm 85_ncc_image_filter 85_ncc_stattics)
    add_executable(${APP} ${APP}.cu)
    set_target_properties(${APP} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(${APP} PUBLIC cublas ${CONAN_LIBS}
            nppc nppif nppisu nppig npps)
endforeach()
