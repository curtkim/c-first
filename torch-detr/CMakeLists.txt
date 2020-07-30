cmake_minimum_required(VERSION 3.15)
project(torch-first LANGUAGES CUDA CXX)

add_definitions("-std=c++17")

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=1")

#find_package(PythonLibs REQUIRED)
#include_directories(${PYTHON_INCLUDE_DIRS})
#target_link_libraries(<your exe or lib> ${PYTHON_LIBRARIES})

include_directories(/usr/local/cuda-10.1/include)
find_package(Torch REQUIRED)

#link_directories( /home/curt/.pyenv/versions/3.7.5/lib/python3.7/site-packages/torch/lib)
#include_directories( /home/curt/.pyenv/versions/3.7.5/lib/python3.7/site-packages/torch/include)

#link_directories( /opt/conda/lib/python3.7/site-packages/torch/lib)
#include_directories( /opt/conda/lib/python3.7/site-packages/torch/include)

message(STATUS "------------------")
foreach(A ${CONAN_LIBS})
	message(STATUS ${A})
endforeach()

foreach(APP main)
	add_executable(${APP} ${APP}.cpp)
	target_link_libraries(${APP} ${TORCH_LIBRARIES} ${CONAN_LIBS})
	target_compile_definitions(${APP} PUBLIC WITH_CUDA)
endforeach()
