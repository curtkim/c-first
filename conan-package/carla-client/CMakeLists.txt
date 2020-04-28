cmake_minimum_required(VERSION 3.15)
project(cmake_wrapper)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

option(LIBCARLA_BUILD_DEBUG "Build debug configuration" OFF)
option(LIBCARLA_BUILD_RELEASE "Build release configuration" ON)
option(LIBCARLA_BUILD_TEST "Build unit tests" OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_CXX_FLAGS "\${CMAKE_CXX_FLAGS} -std=c++17 -pthread" CACHE STRING "" FORCE)

set(libcarla_source_path "${PROJECT_SOURCE_DIR}/source_subfolder/LibCarla/source")
set(libcarla_source_thirdparty_path "${libcarla_source_path}/third-party")

include_directories(${libcarla_source_path})
include_directories(${libcarla_source_thirdparty_path})

if (CARLA_VERSION)
  configure_file(${libcarla_source_path}/carla/Version.h.in ${libcarla_source_path}/carla/Version.h)
endif()

add_subdirectory("source_subfolder/LibCarla/cmake/client")
