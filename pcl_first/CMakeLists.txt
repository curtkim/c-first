cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
project(pcl_first)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

file(COPY sample.ply DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)

foreach(APP pcd_write pcd_convert matrix_transform)
  add_executable(${APP} ${APP}.cpp)
  target_link_libraries(${APP} ${CONAN_LIBS})
endforeach()
