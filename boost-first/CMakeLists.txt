cmake_minimum_required(VERSION 3.15)
project(boost-first)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wformat=0")

foreach(APP
        thread_pool thread_pool2 thread_pool3
        program_options
        date_time
        fiber_first fiber_ping_pong fiber_range_for fiber_adapt_nonblocking fiber_work_sharing
        circular_buffer circular_buffer2
        hana_01 hana_02_serialize hana_03_switch hana_04_integral_constant hana_05_euclidean_dist
        hana02_TypeComputations hana03_Introspection hana03_Introspection_json
        magic_get_test)
    add_executable(${APP} ${APP}.cpp)
    target_link_libraries(${APP} ${CONAN_LIBS})
    target_compile_features(${APP} PUBLIC cxx_std_17)
endforeach()
