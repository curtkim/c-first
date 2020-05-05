CMAKE_MINIMUM_REQUIRED(VERSION 3.15)
PROJECT(static_shared VERSION 1.0.9)

OPTION(BUILD_SHARED_LIBS "Build using shared libraries" ON)

# add the MathFunctions library
ADD_SUBDIRECTORY(MathFunctions)

# add the executable
ADD_EXECUTABLE(main main.cpp)
TARGET_LINK_LIBRARIES(main PUBLIC MathFunctions)
