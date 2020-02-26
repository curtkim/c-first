cmake_minimum_required(VERSION 2.8.12)
project(conan-imgui)

add_definitions("-std=c++17")

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_executable(dcgan
		dcgan.cpp
		)
target_link_libraries(dcgan ${CONAN_LIBS})

add_executable(torchscript
		torchscript.cpp
		)
target_link_libraries(torchscript ${CONAN_LIBS})

