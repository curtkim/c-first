cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
project(pcl_first)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

find_package(OpenMP)

file(COPY sample.ply DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)
file(COPY bun0.pcd DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)
file(COPY bun4.pcd DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)

foreach(APP pcd_write pcd_convert matrix_transform registration_ndt registration_ndt2 vlp_grabber_test )
  add_executable(${APP} ${APP}.cpp)
  target_link_libraries(${APP} ${CONAN_LIBS} OpenMP::OpenMP_CXX)
endforeach()

add_executable(pcap_read pcap_read.cpp)
target_link_libraries(pcap_read PUBLIC pcap)