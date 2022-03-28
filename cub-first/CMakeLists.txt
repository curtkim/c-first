cmake_minimum_required(VERSION 3.16)
project(cub-first LANGUAGES CUDA C CXX)

add_executable(scan_on_device scan_on_device.cu)

set_target_properties(scan_on_device
        PROPERTIES CUDA_ARCHITECTURES 86)

