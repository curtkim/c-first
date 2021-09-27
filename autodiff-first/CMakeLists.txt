cmake_minimum_required(VERSION 3.15)
project(sophus-first)

set ( CMAKE_EXPORT_COMPILE_COMMANDS on )

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

foreach(APP
        01_simple
        11_forward_single_variable
        12_foward_multi_variable
        13_forward-multi-variable-function-with-parameters
        14_forward-higher-order-directional-derivatives

        31_backward_single_variable
        )
    add_executable(${APP} ${APP}.cpp)
    target_compile_features(${APP} PUBLIC cxx_std_17)
    target_link_libraries(${APP} PUBLIC ${CONAN_LIBS})
endforeach()
