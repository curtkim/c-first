cmake_minimum_required(VERSION 3.15)
project(vulkan)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

find_package(Vulkan)

foreach(APP main 13_framebuffers 16_swap_chain_recreation)
    add_executable(${APP} ${APP}.cpp)
    target_link_libraries(${APP} ${CONAN_LIBS} X11 Vulkan::Vulkan)
    target_compile_features(${APP} PUBLIC cxx_std_17)
endforeach()