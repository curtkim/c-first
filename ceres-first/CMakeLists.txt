cmake_minimum_required (VERSION 3.0)
project(ceres-first)

# main
add_executable(helloworld helloworld.cc)
add_executable(curve_fitting curve_fitting.cc)



# lib
find_package(glog 0.4.0 REQUIRED)
target_link_libraries(helloworld glog::glog)
target_link_libraries(curve_fitting glog::glog)

find_package(ceres 1.14.0 REQUIRED)
target_link_libraries(helloworld ceres)
target_link_libraries(curve_fitting ceres)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
target_link_libraries(helloworld Eigen3::Eigen)
target_link_libraries(curve_fitting Eigen3::Eigen)
