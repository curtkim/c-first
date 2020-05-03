cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
project(pcl_first)

#find_package(PCL 1.3 REQUIRED COMPONENTS common io)
#include_directories(${PCL_INCLUDE_DIRS})
#link_directories(${PCL_LIBRARY_DIRS})
#add_definitions(${PCL_DEFINITIONS})

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

add_executable(pcd_write_test pcd_write.cpp)
target_link_libraries(pcd_write_test ${CONAN_LIBS})

add_executable(pcd_convert pcd_convert.cpp)
target_link_libraries(pcd_convert ${CONAN_LIBS})
