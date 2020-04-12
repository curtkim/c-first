cmake_minimum_required(VERSION 3.10)
project (Tutorial VERSION 1.0.2)

# configure a header file to pass some of the CMake settings
# to the source code
configure_file (
  "${PROJECT_SOURCE_DIR}/TutorialConfig.h.in"
  "${PROJECT_BINARY_DIR}/TutorialConfig.h"
  )

message(STATUS "PROJECT_SOURCE_DIR=${PROJECT_SOURCE_DIR}")
message(STATUS "PROJECT_SOURCE_DIR=${PROJECT_BINARY_DIR}")
message(STATUS "CMAKE_CURRENT_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}")

# add the executable
add_executable(Tutorial tutorial.cxx)

target_compile_features(Tutorial PUBLIC cxx_std_11)

# add the binary tree to the search path for include files
# so that we will find TutorialConfig.h
target_include_directories(Tutorial PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<INSTALL_INTERFACE:include/step1>  # <prefix>/include/mylib
)
