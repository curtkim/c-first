cmake_minimum_required(VERSION 3.16)
project(continuable-first)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

foreach(APP main example-asio ) #with_co_await
    add_executable(${APP} ${APP}.cpp)
    target_link_libraries(${APP} ${CONAN_LIBS})
    target_compile_features(${APP} PUBLIC cxx_std_20)
endforeach()

#target_compile_definitions(with_co_await PUBLIC CONTINUABLE_WITH_EXPERIMENTAL_COROUTINE)