cmake_minimum_required(VERSION 3.15)
project(carla-frist)

set(CMAKE_CXX_STANDARD 17)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

foreach(APP
    save_image
    save_lidar
    rxcpp_sensor_create
    rxcpp_sensor_all)
  add_executable(${APP} ${APP}.cpp)
  target_link_libraries(${APP} ${CONAN_LIBS})
  target_compile_features(${APP} PRIVATE cxx_std_17)
endforeach()


file(COPY Town01.xodr DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)

add_executable(xodr2geojson_convert xodr2geojson_convert.cpp xodr2geojson.cpp)
target_link_libraries(xodr2geojson_convert ${CONAN_LIBS})
target_compile_features(xodr2geojson_convert PRIVATE cxx_std_17)


add_executable(opengl_camera opengl_camera)
target_link_libraries(opengl_camera ${CONAN_LIBS} ${OPENGL_LIBRARY} X11)
target_compile_features(opengl_camera PRIVATE cxx_std_17)
