add_library(MathFunctions mysqrt.cxx)
target_include_directories(MathFunctions PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
)
