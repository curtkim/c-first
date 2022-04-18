cmake_minimum_required(VERSION 3.20)
project(openvdb-first LANGUAGES CXX)


include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()


#file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/logo.png
#    DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)

foreach (APP 01_hello 02_create_write_grid 21_hello_point)
	add_executable(${APP} ${APP}.cpp)
    target_link_libraries(${APP} PUBLIC ${CONAN_LIBS})
endforeach ()

