cmake_minimum_required(VERSION 3.15)
project(conan-opengl)

set(CMAKE_VERBOSE_MAKEFILE ON)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/app.cfg DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)

foreach(APP
        first scene display plot
        )
  add_executable(${APP} ${APP}.cpp) # memory_tracking.cpp
  target_link_libraries(${APP} ${CONAN_LIBS} X11)
  target_compile_features(${APP} PUBLIC cxx_std_17)
  #target_compile_definitions(${APP} PUBLIC STB_IMAGE_IMPLEMENTATION)
endforeach()
