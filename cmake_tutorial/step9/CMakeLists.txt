CMAKE_MINIMUM_REQUIRED(VERSION 3.10)
PROJECT(Tutorial VERSION 1.0.9)

OPTION(BUILD_SHARED_LIBS "Build using shared libraries" ON)

# configure a header file to pass the version number only
CONFIGURE_FILE(
  "${PROJECT_SOURCE_DIR}/TutorialConfig.h.in"
  "${PROJECT_BINARY_DIR}/TutorialConfig.h"
)

# add the MathFunctions library
ADD_SUBDIRECTORY(MathFunctions)

# add the executable
ADD_EXECUTABLE(Tutorial tutorial.cxx)
TARGET_LINK_LIBRARIES(Tutorial PUBLIC MathFunctions)
TARGET_INCLUDE_DIRECTORIES(Tutorial PUBLIC
  "${PROJECT_BINARY_DIR}"
  )