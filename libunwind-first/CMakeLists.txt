cmake_minimum_required(VERSION 3.15)
project(libunwind_first CXX C)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

foreach(A ${CONAN_LIBS})
    message(STATUS ${A})
endforeach()

foreach(APP
        01_first)
    add_executable(${APP} ${APP}.c)
    target_link_libraries(${APP} PUBLIC ${CONAN_LIBS})
    set_target_properties(${APP} PROPERTIES CXX_STANDARD 17)
endforeach()
