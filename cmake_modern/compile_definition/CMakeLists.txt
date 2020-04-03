cmake_minimum_required(VERSION 3.8)
project(CompileDefinition VERSION 0.0.1)

add_executable(main main.cpp)
target_compile_features(main PUBLIC cxx_std_17)
target_compile_definitions(main PRIVATE FOO_SIZE=10)
