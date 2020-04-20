cmake_minimum_required(VERSION 3.15)
project(asio)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

# TODO
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
#add_executable(range_based_for range_based_for.cpp)
#target_compile_features(range_based_for PUBLIC cxx_std_20)
#target_compile_options(range_based_for PRIVATE -fcoroutines-ts)
#target_compile_definitions(range_based_for PRIVATE ASIO_HAS_CO_AWAIT)
#target_link_libraries(range_based_for ${CONAN_LIBS})

foreach(APP
    stdout_write file_write_future file_write_c
    timer daytime_client)
  add_executable(${APP} ${APP}.cpp)
  target_link_libraries(${APP} ${CONAN_LIBS})
endforeach()
