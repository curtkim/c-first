cmake_minimum_required(VERSION 3.15)
project(carla-frist)

set(CMAKE_CXX_STANDARD 17)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

foreach(APP
    01_save_image
    02_save_lidar
    11_rxcpp_sensor_create
    12_rxcpp_sensor_all)
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

add_executable(91_opengl_camera 91_opengl_camera)
target_link_libraries(91_opengl_camera ${CONAN_LIBS} ${OPENGL_LIBRARY} X11)
target_compile_features(91_opengl_camera PRIVATE cxx_std_17)
