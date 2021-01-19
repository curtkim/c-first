cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
project(pcl_first)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

foreach(APP 01_first )
  add_executable(${APP} ${APP}.cpp)
  target_link_libraries(${APP} ${CONAN_LIBS})
  #target_compile_features(${APP} PUBLIC cxx_std_20)
endforeach()
