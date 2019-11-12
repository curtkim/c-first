cmake_minimum_required (VERSION 3.2 FATAL_ERROR)
project (modern-cpp14)

set (CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(test_vector test_vector.cpp)
set_property(TARGET test_vector PROPERTY CXX_STANDARD 14)

add_executable(test_shared_ptr test_shared_ptr.cpp )
set_property(TARGET test_shared_ptr PROPERTY CXX_STANDARD 14)

add_executable(test_shared_ptr2 test_shared_ptr2.cpp )
set_property(TARGET test_shared_ptr2 PROPERTY CXX_STANDARD 14)
target_link_libraries( test_shared_ptr2 pthread )
