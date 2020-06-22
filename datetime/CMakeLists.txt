cmake_minimum_required (VERSION 3.2 FATAL_ERROR)
project (datetime)

#set (CMAKE_CXX_STANDARD_REQUIRED ON)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

foreach(APP main format execution_time time_point)
    add_executable(${APP} ${APP}.cpp)
    set_property(TARGET ${APP} PROPERTY CXX_STANDARD 17)
    target_link_libraries(${APP} PUBLIC ${CONAN_LIBS})
    #target_link_libraries( modern-thread pthread )
endforeach()
