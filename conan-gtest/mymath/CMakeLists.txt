project(mymath)

set(HEADER_FILES
        mymath.hpp
        )

set(SOURCE_FILES
        mymath.cpp
        )

add_library(mymath STATIC ${SOURCE_FILES} ${HEADER_FILES})

add_executable(mymath_test mymath_test.cpp)
target_link_libraries(mymath_test mymath ${CONAN_LIBS})