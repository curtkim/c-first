# add the library that runs
ADD_LIBRARY(MathFunctions MathFunctions.cpp)

# state that anybody linking to us needs to include the current source dir
# to find MathFunctions.h, while we don't.
TARGET_INCLUDE_DIRECTORIES(MathFunctions
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
)


# install rules
INSTALL(TARGETS MathFunctions DESTINATION lib)
INSTALL(FILES MathFunctions.h DESTINATION include)