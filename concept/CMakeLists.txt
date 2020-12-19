cmake_minimum_required (VERSION 3.15 FATAL_ERROR)
project (concept-cpp)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

foreach(APP
        01_main 02_test-predicate 03_norm 04_tick_tok_with_hana 05_shape_as_interface
        11_std_regular 12_not_copyable
        )
    message("${APP}")
    add_executable(${APP} ${APP}.cpp)
    target_compile_features(${APP} PUBLIC cxx_std_20)
    target_link_libraries(${APP} PUBLIC ${CONAN_LIBS})
    #set_property(TARGET ${APP} PROPERTY CXX_STANDARD 20)
endforeach()
