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
    31_rxcpp_image_one_file)
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

foreach(APP 91_opengl_camera 92_opengl_lidar)
  add_executable(${APP} ${APP}.cpp)
  target_link_libraries(${APP} ${CONAN_LIBS} ${OPENGL_LIBRARY} X11)
  target_compile_features(${APP} PUBLIC cxx_std_17)
endforeach()