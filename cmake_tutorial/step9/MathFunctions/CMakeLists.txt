# add the library that runs
ADD_LIBRARY(MathFunctions MathFunctions.cxx)

# state that anybody linking to us needs to include the current source dir
# to find MathFunctions.h, while we don't.
TARGET_INCLUDE_DIRECTORIES(MathFunctions
  INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}
  )

# should we use our own math functions
OPTION(USE_MYMATH "Use tutorial provided math implementation" ON)
IF (USE_MYMATH)

  TARGET_COMPILE_DEFINITIONS(MathFunctions PRIVATE "USE_MYMATH")

  # first we add the executable that generates the table
  ADD_EXECUTABLE(MakeTable MakeTable.cxx)

  # add the command to generate the source code
  ADD_CUSTOM_COMMAND(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/Table.h
    COMMAND MakeTable ${CMAKE_CURRENT_BINARY_DIR}/Table.h
    DEPENDS MakeTable
  )

  # library that just does sqrt
  ADD_LIBRARY(SqrtLibrary STATIC
    mysqrt.cxx
    ${CMAKE_CURRENT_BINARY_DIR}/Table.h
    )

  # state that we depend on our binary dir to find Table.h
  TARGET_INCLUDE_DIRECTORIES(SqrtLibrary PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}
    )

  # state that SqrtLibrary need PIC when the default is shared libraries
  SET_TARGET_PROPERTIES(SqrtLibrary PROPERTIES
    POSITION_INDEPENDENT_CODE ${BUILD_SHARED_LIBS}
    )

  TARGET_LINK_LIBRARIES(MathFunctions PRIVATE SqrtLibrary)
ENDIF ()

# define the symbol stating we are using the declspec(dllexport) when
# building on windows
TARGET_COMPILE_DEFINITIONS(MathFunctions PRIVATE "EXPORTING_MYMATH")

# install rules
INSTALL(TARGETS MathFunctions DESTINATION lib)
INSTALL(FILES MathFunctions.h DESTINATION include)