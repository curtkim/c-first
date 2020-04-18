cmake_minimum_required(VERSION 3.10)
project(rxterm-first VERSION 1.0.2)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()


add_executable(raw raw.cpp)

add_executable(main main.cpp)
target_compile_features(main PUBLIC cxx_std_17)
target_include_directories(main PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  )


add_executable(counter counter.cpp)
target_compile_features(counter PUBLIC cxx_std_17)
target_include_directories(counter PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  )


add_executable(rxcpp_progress rxcpp_progress.cpp)
target_compile_features(rxcpp_progress PUBLIC cxx_std_17)
target_include_directories(rxcpp_progress PRIVATE
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  )
