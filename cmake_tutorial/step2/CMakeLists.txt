cmake_minimum_required(VERSION 3.10)
project (Tutorial VERSION 1.0.2)

# should we use our own math functions?
option (USE_MYMATH
        "Use tutorial provided math implementation" ON)

configure_file (
  "${PROJECT_SOURCE_DIR}/TutorialConfig.h.in"
  "${PROJECT_BINARY_DIR}/TutorialConfig.h"
)

# add the MathFunctions library?
if (USE_MYMATH)
  include_directories ("${PROJECT_SOURCE_DIR}/MathFunctions")
  add_subdirectory (MathFunctions)
  set (EXTRA_LIBS ${EXTRA_LIBS} MathFunctions)
endif (USE_MYMATH)

# add the executable
add_executable (Tutorial tutorial.cxx)

target_include_directories(Tutorial PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  $<INSTALL_INTERFACE:include/step2>  # <prefix>/include/mylib
)
target_link_libraries (Tutorial ${EXTRA_LIBS})