cmake_minimum_required(VERSION 3.15)
project(cilantro-first)

set(CMAKE_VERBOSE_MAKEFILE ON)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

find_package(OpenMP)

foreach(FILE frame_1.ply frame_2.ply test.ply)
  file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/samples/${FILE} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)
endforeach()

foreach(APP
  convex_hull
        )
  add_executable(${APP} ${APP}.cpp)
  target_link_libraries(${APP} ${CONAN_LIBS} X11 OpenMP::OpenMP_CXX)
  target_compile_features(${APP} PUBLIC cxx_std_17)
  #target_compile_definitions(${APP} PUBLIC STB_IMAGE_IMPLEMENTATION)
endforeach()
