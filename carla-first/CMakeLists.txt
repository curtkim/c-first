cmake_minimum_required(VERSION 3.15)
project(carla-frist)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

foreach(APP
    01_save_image
    02_save_lidar
    03_multi_camera_performance
    11_rxcpp_sensor_create
    12_rxcpp_sensor_all
    31_rxcpp_image_one_file
    61_sensor_server
    62_sensor_client 62_sensor_client_rx
    71_sensor_server
    71_sensor_server_rx
    71_sensor_server_dummy
    72_sensor_client_sync_rx
    72_sensor_client_dummy
    81_sensor_server
    )
  add_executable(${APP} ${APP}.cpp)
  target_link_libraries(${APP} ${CONAN_LIBS})
  target_compile_features(${APP} PRIVATE cxx_std_17)
endforeach()

file(COPY Town01.xodr DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)

foreach(APP
        21_xodr2geojson_convert
        22_traffic_light_info)
  add_executable(${APP} ${APP}.cpp xodr2geojson.cpp utils.cpp)
  target_link_libraries(${APP} ${CONAN_LIBS})
  target_compile_features(${APP} PRIVATE cxx_std_17)
endforeach()

# opengl
foreach(APP
        51_camera_opengl
        52_lidar_opengl
        62_sensor_client_sync
        72_sensor_client_sync_opengl)
  add_executable(${APP} ${APP}.cpp)
  target_link_libraries(${APP} ${CONAN_LIBS} ${OPENGL_LIBRARY} X11)
  target_compile_features(${APP} PUBLIC cxx_std_17)
endforeach()