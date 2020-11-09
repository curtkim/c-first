cmake_minimum_required (VERSION 3.15 FATAL_ERROR)
project (meta-first )

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

foreach(APP 01_add 02_relaxNG 03_decay 04_enable_if 05_is_lvalue_reference
        11_generic_copy 12_tag_dispatching)
    add_executable(${APP} ${APP}.cpp)
    #target_link_libraries(${APP} uring )
    #target_compile_definitions(${APP} PUBLIC _GNU_SOURCE)
    target_compile_features(${APP} PUBLIC cxx_std_17)
endforeach()


add_executable(with_dsl dsl/main.cpp )
target_compile_features(with_dsl PUBLIC cxx_std_17)

add_executable(reactive reactive/main.cpp reactive/service.cpp)
target_link_libraries(reactive PUBLIC pthread)
target_compile_features(reactive PUBLIC cxx_std_17)
