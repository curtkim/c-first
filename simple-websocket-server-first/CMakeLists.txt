cmake_minimum_required(VERSION 3.15)
project(simple-websocket-server)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

foreach(APP ws_server ws_client ws_timer_server ws_client_forever ws_client_asio)
    add_executable(${APP} ${APP}.cpp)
    target_link_libraries(${APP} PUBLIC ${CONAN_LIBS})
    target_compile_features(${APP} PUBLIC cxx_std_17)
endforeach()
