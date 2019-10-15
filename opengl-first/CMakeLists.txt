cmake_minimum_required (VERSION 3.2 FATAL_ERROR)
project (opengl-first)

set (CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenGL REQUIRED)

# Compile external dependencies
add_subdirectory (external)

include_directories(
        external/glfw-3.1.2/include/
        external/glm-0.9.7.1/
        external/glew-1.13.0/include/
        .
)

set(ALL_LIBS
        ${OPENGL_LIBRARY}
        glfw
        GLEW_1130
        )

# add the executable
add_executable(opengl-first
        main.cpp
        common/shader.cpp
        common/shader.hpp
        SimpleFragmentShader.fragmentshader
        SimpleVertexShader.vertexshader
        )
set_property(TARGET opengl-first PROPERTY CXX_STANDARD 17)
target_link_libraries (opengl-first ${ALL_LIBS})
