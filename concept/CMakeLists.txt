cmake_minimum_required (VERSION 3.15 FATAL_ERROR)
project (concept-cpp)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

foreach(APP
        main range_in test-predicate norm tick_tok shape
        )
    message("${APP}")
    add_executable(${APP} ${APP}.cpp)
    target_compile_features(${APP} PUBLIC cxx_std_20)
    target_link_libraries(${APP} PUBLIC ${CONAN_LIBS})
    #set_property(TARGET ${APP} PROPERTY CXX_STANDARD 20)
endforeach()
