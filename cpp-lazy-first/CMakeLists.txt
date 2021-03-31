cmake_minimum_required(VERSION 3.16)
project(continuable-first)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

foreach(APP main filter group_by  )
    add_executable(${APP} ${APP}.cpp)
    target_link_libraries(${APP} ${CONAN_LIBS})
    target_compile_features(${APP} PUBLIC cxx_std_17)
endforeach()
