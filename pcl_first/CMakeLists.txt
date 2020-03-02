cmake_minimum_required(VERSION 2.6 FATAL_ERROR)
project(pcl_first)

find_package(PCL 1.3 REQUIRED COMPONENTS common io)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(pcd_write_test pcd_write.cpp)
target_link_libraries(pcd_write_test ${PCL_LIBRARIES})

add_executable(pcd_convert pcd_convert.cpp)
target_link_libraries(pcd_convert ${PCL_LIBRARIES})
