cmake_minimum_required(VERSION 3.15)
project(conan-opengl)

find_package(OpenGL REQUIRED)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

foreach(FILE
        SimpleFragmentShader.fragmentshader
        SimpleVertexShader.vertexshader
        bunny.off
        lidar.ply
        carla_town05_link.shp
        carla_town05_link.shx
        00000_camera0.png
        00000_camera1.png
        00000_lidar.ply)
  file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/resources/${FILE} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)
endforeach()

foreach(APP
        00_glm_quantinion
        01_triangle
        01_triangle_curt_style
        03_texture_two
        04_texture_from_cuda_mem
        12_point 13_point_multi 14_grid 15_polylines 16_triangle_animation
        21_renderable
        31_mouse
        32_mouse_map_control_wo_camera
        41_load-model-libigl
        42_load_ply
        51_shapefile
        61_opengl_thread
        62_thread_render_loop
        90_map_buffer
        90_without_cuda
        )
  add_executable(${APP} ${APP}.cpp common/shader.cpp ) # memory_tracking.cpp
  target_link_libraries(${APP} ${CONAN_LIBS} ${OPENGL_LIBRARY} X11)
  target_compile_features(${APP} PUBLIC cxx_std_17)
  target_compile_definitions(${APP} PUBLIC STB_IMAGE_IMPLEMENTATION)
endforeach()

add_executable(02_second_grid_with_camera 02_second_grid_with_camera.cpp common/shader.cpp common/camera2.cpp)
target_link_libraries(02_second_grid_with_camera ${CONAN_LIBS} ${OPENGL_LIBRARY} X11)
target_compile_features(02_second_grid_with_camera PUBLIC cxx_std_17)
target_compile_definitions(02_second_grid_with_camera PUBLIC STB_IMAGE_IMPLEMENTATION)

foreach(APP 71_egl 71_egl_triangle 72_egl_texture_and_box 73_egl_geometry_shader)
  add_executable(${APP} ${APP}.cpp)
  target_link_libraries(${APP} EGL ${CONAN_LIBS} ${OPENGL_LIBRARY})
  target_compile_features(${APP} PUBLIC cxx_std_17)
  target_compile_definitions(${APP} PUBLIC STB_IMAGE_IMPLEMENTATION)
endforeach()