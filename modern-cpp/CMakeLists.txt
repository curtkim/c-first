cmake_minimum_required (VERSION 3.2 FATAL_ERROR)
project (modern-cpp)

set (CMAKE_CXX_STANDARD_REQUIRED ON)

### 11

add_executable(array array.cpp)
set_property(TARGET array PROPERTY CXX_STANDARD 11)

add_executable(thread_pool_executor_example thread_pool_executor_example.cpp thread_pool_executor.hpp concurrent_queue.hpp)
set_property(TARGET thread_pool_executor_example PROPERTY CXX_STANDARD 11)
target_link_libraries( thread_pool_executor_example pthread )


### 14

add_executable(test_vector test_vector.cpp)
set_property(TARGET test_vector PROPERTY CXX_STANDARD 14)

add_executable(test_shared_ptr test_shared_ptr.cpp)
set_property(TARGET test_shared_ptr PROPERTY CXX_STANDARD 14)

add_executable(test_shared_ptr2 test_shared_ptr2.cpp)
set_property(TARGET test_shared_ptr2 PROPERTY CXX_STANDARD 14)
target_link_libraries( test_shared_ptr2 pthread )

add_executable(generic generic.cpp)
set_property(TARGET generic PROPERTY CXX_STANDARD 14)
target_link_libraries( generic pthread )

add_executable(Threadpool_test Threadpool_test.cpp Threadpool.hpp)
set_property(TARGET Threadpool_test PROPERTY CXX_STANDARD 14)
target_link_libraries( Threadpool_test pthread )

add_executable(rvalue_reference rvalue_reference.cpp)
set_property(TARGET rvalue_reference PROPERTY CXX_STANDARD 14)


### 17

add_executable(modern-thread thread_main.cpp)
set_property(TARGET modern-thread PROPERTY CXX_STANDARD 17)
target_link_libraries( modern-thread pthread )

foreach(APP
        main
        cpp17
        move_forward
        test_string_view
        test_string_view2
        test_string_view3)
    message("${APP}")
    add_executable(${APP} ${APP}.cpp)
    set_property(TARGET ${APP} PROPERTY CXX_STANDARD 17)
endforeach()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    add_executable(parallel_transform parallel_transform.cpp)
    set_property(TARGET parallel_transform PROPERTY CXX_STANDARD 17)
    target_link_libraries(parallel_transform PUBLIC OpenMP::OpenMP_CXX)
endif()