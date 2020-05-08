cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
project(pcl_first)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

find_package(OpenMP)

file(COPY sample.ply DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)
file(COPY bun0.pcd DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)
file(COPY bun4.pcd DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)

foreach(APP pcd_write pcd_convert matrix_transform registration_ndt registration_ndt2)
  add_executable(${APP} ${APP}.cpp)
  target_link_libraries(${APP} ${CONAN_LIBS} OpenMP::OpenMP_CXX)
endforeach()
