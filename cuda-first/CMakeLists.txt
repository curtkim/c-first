cmake_minimum_required(VERSION 3.16)
project(cuda-first LANGUAGES CUDA C CXX)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_executable(first first.cc first-gpu.cu)

add_executable(nvtx_first nvtx_first.cpp)
target_include_directories(nvtx_first PUBLIC /usr/local/cuda/include)
target_link_directories(nvtx_first PUBLIC /usr/local/cuda/lib64)
target_link_libraries(nvtx_first PUBLIC nvToolsExt)

add_executable(do_cpu do_cpu.cpp)
add_executable(do_gpu do_gpu.cu)
add_executable(cuda_struct cuda_struct.cu)

add_executable(cublas_mm cublas_mm.cu)
target_link_libraries(cublas_mm cublas)

add_executable(matrix_multiplication_by_rand matrix_multiplication_by_rand.cu)
target_link_libraries(matrix_multiplication_by_rand cublas curand)

include_directories(common)

add_executable(streamOrderedAllocation streamOrderedAllocation.cu)
add_executable(simpleCallback 1_simple/simpleCallback.cu 1_simple/multithreading.cpp)

foreach(APP asyncAPI simpleCudaGraph simpleStream )
    add_executable(${APP} 1_simple/${APP}.cu)
endforeach()

foreach(APP simpleGL)
    add_executable(${APP} 2_graphic/${APP}.cu)
    target_link_libraries(${APP} PUBLIC glut GL GLU)
endforeach()

add_executable(opengl_cuda_vertex_interop 2_my_graphic/opengl_cuda_vertex_interop.cpp common/shader.cpp 2_graphic/simple_kernel.cu)
target_link_libraries(opengl_cuda_vertex_interop PUBLIC ${CONAN_LIBS} ${OPENGL_LIBRARY} X11 nvToolsExt)
target_include_directories(opengl_cuda_vertex_interop PRIVATE /usr/local/cuda/include)
target_compile_features(opengl_cuda_vertex_interop PUBLIC cxx_std_17)


add_executable(postProcessGL 3_image/postProcessGL.cpp 3_image/postProcessGL_kernel.cu)
target_link_libraries(postProcessGL PUBLIC ${CONAN_LIBS} ${OPENGL_LIBRARY} X11 nvToolsExt)
target_include_directories(postProcessGL PRIVATE /usr/local/cuda/include)
target_compile_features(postProcessGL PUBLIC cxx_std_17)


foreach(APP cannyEdgeDetect freeImageInteropNPP jpegNPP)
    add_executable(${APP} 7_library/${APP}.cpp)
    #target_link_libraries(${APP} PUBLIC glut GL GLU)
    target_include_directories(${APP} PRIVATE /usr/local/cuda/include)
    target_link_directories(${APP} PRIVATE /usr/local/cuda/lib64)
    target_link_libraries(${APP} PUBLIC ${CONAN_LIBS})
    target_link_libraries(${APP} PRIVATE cuda cudart
            nppc nppicom nppig nppisu nppial nppidei nppim nppitc nppicc nppif nppist)
            #nppc nppisu nppif ) # nppc nppicom nppig nppisu nppial nppidei nppim nppitc nppicc nppif nppist
endforeach()


add_executable(simpleTexture3D 2_graphic/simpleTexture3D.cpp 2_graphic/simpleTexture3D_kernel.cu)
target_link_libraries(simpleTexture3D PUBLIC glut GL GLU)
target_include_directories(simpleTexture3D PRIVATE /usr/local/cuda/include)
file(COPY 2_graphic/Bucky.raw DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

find_package(Vulkan)
add_executable(vulkanCUDASinewave 2_graphic/vulkanCUDASinewave.cu)
target_link_libraries(vulkanCUDASinewave ${CONAN_LIBS} X11 Vulkan::Vulkan)
target_compile_features(vulkanCUDASinewave PUBLIC cxx_std_17)
file(COPY 2_graphic/shader_sine.frag DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY 2_graphic/shader_sine.vert DESTINATION ${CMAKE_CURRENT_BINARY_DIR})


add_executable(oceanFFT 5_simulation/oceanFFT.cpp 5_simulation/oceanFFT_kernel.cu)
target_link_libraries(oceanFFT PUBLIC cufft glut GL GLU)
target_include_directories(oceanFFT PRIVATE /usr/local/cuda/include)
file(COPY 5_simulation/ocean.frag DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
file(COPY 5_simulation/ocean.vert DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_executable(template_function_pointer template_function_pointer.cu)


foreach(APP practice_01_cuda_malloc practice_02_nppi_malloc practice_03_cuda_memcpy2d practice_04_cudaMemcpy2D_crop)
    add_executable(${APP} 0_malloc/${APP}.c)
    target_include_directories(${APP} PRIVATE /usr/local/cuda/include)
    target_link_directories(${APP} PRIVATE /usr/local/cuda/lib64)
    target_link_libraries(${APP} PUBLIC ${CONAN_LIBS})
    target_link_libraries(${APP} PRIVATE cuda cudart nppisu)
endforeach()

foreach(APP npp_min_max)
    add_executable(${APP} ${APP}.cpp)
    target_include_directories(${APP} PRIVATE /usr/local/cuda/include)
    target_link_directories(${APP} PRIVATE /usr/local/cuda/lib64)
    target_link_libraries(${APP} PUBLIC ${CONAN_LIBS})
    target_link_libraries(${APP} PRIVATE cuda cudart npps nppist)
endforeach()



set_target_properties(first
        nvtx_first
        do_cpu do_gpu cuda_struct
        cublas_mm
        matrix_multiplication_by_rand
        streamOrderedAllocation
        simpleCallback
        asyncAPI simpleCudaGraph simpleStream
        simpleGL
        postProcessGL
        opengl_cuda_vertex_interop
        cannyEdgeDetect freeImageInteropNPP jpegNPP
        simpleTexture3D
        vulkanCUDASinewave
        oceanFFT
        template_function_pointer
        practice_01_cuda_malloc practice_02_nppi_malloc practice_03_cuda_memcpy2d practice_04_cudaMemcpy2D_crop

        PROPERTIES CUDA_ARCHITECTURES 61)