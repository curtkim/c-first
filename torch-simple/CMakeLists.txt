cmake_minimum_required(VERSION 3.16)
project(torch-simple LANGUAGES CXX)

set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

#find_package(Torch REQUIRED PATHS "/home/curt/projects/c-first/conan-package/torch_prebuilt/libtorch")
find_package(Torch REQUIRED PATHS "/home/curt/.conan/data/torch/1.8.1/curt/prebuilt/package/156b4f8deb813b75883a341f019bbe9e947cc3f9")

foreach(ITEM ${TORCH_LIBRARIES})
	message( ${ITEM} )
endforeach()

# torch
# torch_library
# /home/curt/projects/c-first/conan-package/torch_prebuilt/libtorch/lib/libc10.so
# /home/curt/projects/c-first/conan-package/torch_prebuilt/libtorch/lib/libc10_cuda.so
# /usr/local/cuda/lib64/stubs/libcuda.so
# /usr/local/cuda/lib64/libnvrtc.so
# /usr/local/cuda/lib64/libnvToolsExt.so
# /usr/local/cuda/lib64/libcudart.so

foreach(APP main)
	add_executable(${APP} ${APP}.cpp)
	target_link_libraries(${APP} ${TORCH_LIBRARIES})
	target_compile_features(${APP} PUBLIC cxx_std_17)
endforeach()
