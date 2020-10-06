cmake_minimum_required(VERSION 3.15)
project(cuda-first LANGUAGES CUDA CXX)

add_executable(cuda-version main.cc gpu.cu)

add_executable(do_cpu do_cpu.cpp)
add_executable(do_gpu do_gpu.cu)

add_executable(cublas_mm cublas_mm.cu)
target_link_libraries(cublas_mm cublas)

add_executable(matrix_multiplication_by_rand matrix_multiplication_by_rand.cu)
target_link_libraries(matrix_multiplication_by_rand cublas curand)

include_directories(common)

foreach(APP asyncAPI simpleCudaGraph simpleStream)
    add_executable(${APP} 1_simple/${APP}.cu)
endforeach()

foreach(APP simpleGL)
    add_executable(${APP} 2_graphic/${APP}.cu)
    target_link_libraries(${APP} PUBLIC glut GL GLU)
endforeach()

add_executable(simpleTexture3D 2_graphic/simpleTexture3D.cpp 2_graphic/simpleTexture3D_kernel.cu)
target_link_libraries(simpleTexture3D PUBLIC glut GL GLU)
target_include_directories(simpleTexture3D PRIVATE /usr/local/cuda/include)
file(COPY 2_graphic/Bucky.raw DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_executable(oceanFFT 5_simulation/oceanFFT.cpp 5_simulation/oceanFFT_kernel.cu)
target_link_libraries(oceanFFT PUBLIC cufft glut GL GLU)
target_include_directories(oceanFFT PRIVATE /usr/local/cuda/include)
file(COPY 5_simulation/ocean.frag DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY 5_simulation/ocean.vert DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
