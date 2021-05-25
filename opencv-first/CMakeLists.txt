cmake_minimum_required(VERSION 3.16)
project(opencv-first)

set(CMAKE_VERBOSE_MAKEFILE ON)
#set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

#TODO
link_directories( /usr/local/cuda/lib64)

foreach(DEP ${CONAN_LIBS})
  message(${DEP})
endforeach()

foreach(APP
    01_ui
    02_cvtColor
    11_mat 12_scan 13_sobel
    50_epipolar_lines 51_contours2 52_delaunay2 53_detect_mser 54_digits_svm 55_kalman 56_kmeans 57_fitellipse
    58_flann_search_dataset)
  add_executable(${APP} ${APP}.cpp)
  target_link_libraries(${APP} ${CONAN_LIBS})
  target_compile_features(${APP} PRIVATE cxx_std_17)
endforeach()

# opengl
foreach(APP 80_show_opencv_image 81_show_opencv_video 82_jpeg2mkv)
  add_executable(${APP} ${APP}.cpp)
  target_link_libraries(${APP} ${CONAN_LIBS} nvrtc cudart X11 )
  target_compile_features(${APP} PRIVATE cxx_std_17)
endforeach()

# gpu
#foreach(APP
#        90_gpumat)
#  add_executable(${APP} ${APP}.cpp)
#  target_link_libraries(${APP} ${CONAN_LIBS}
#          nvrtc cudart
#          nppc nppig nppisu nppial nppidei nppim nppitc nppicc nppif nppist)
#  target_compile_features(${APP} PRIVATE cxx_std_17)
#endforeach()

