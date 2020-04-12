cmake_minimum_required(VERSION 3.10)
project (Tutorial VERSION 1.0.3)

# should we use our own math functions?
option (USE_MYMATH
  "Use tutorial provided math implementation" ON)

configure_file (
  "${PROJECT_SOURCE_DIR}/TutorialConfig.h.in"
  "${PROJECT_BINARY_DIR}/TutorialConfig.h"
)

if (USE_MYMATH)
  add_subdirectory (MathFunctions)
  set (EXTRA_LIBS ${EXTRA_LIBS} MathFunctions)
endif (USE_MYMATH)


add_executable (Tutorial tutorial.cxx)
target_link_libraries (Tutorial ${EXTRA_LIBS})
target_include_directories(Tutorial PUBLIC
  "${PROJECT_BINARY_DIR}"
)