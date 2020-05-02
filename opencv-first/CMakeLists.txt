cmake_minimum_required(VERSION 2.8.12)
project(MD5Encrypter)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

foreach(APP lena sobel
    01_ui
    11_mat 12_scan)
  add_executable(${APP} ${APP}.cpp)
  target_link_libraries(${APP} ${CONAN_LIBS})
  target_compile_features(${APP} PRIVATE cxx_std_17)
endforeach()
