
# Required CMake
cmake_minimum_required(VERSION 3.16)
project(cpp_modules_example)

# Include file with required functions
include(CXXModules.cmake)

# Use special function for creating C++ modules library.
# Same as add_library but also creates interface files
# and add required flags for current compiler
add_module_library(hello_world
    hello.cpp
    src/world.cpp
)

# Create executable target
add_module_executable(${PROJECT_NAME}
    main.cpp
)

# Link C++ modules library to the executable
target_link_module_libraries(${PROJECT_NAME}
    hello_world
)
