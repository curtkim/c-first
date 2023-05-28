cmake_minimum_required(VERSION 3.25)
project(static_assert_first)

set(CMAKE_CXX_STANDARD 20)

add_executable(static_assert_first main.cpp)
target_compile_options(static_assert_first PRIVATE -fimplicit-constexpr)
