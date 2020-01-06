cmake_minimum_required (VERSION 3.2 FATAL_ERROR)
project (modern-cpp)

set (CMAKE_CXX_STANDARD_REQUIRED ON)

add_executable(modern-cpp main.cpp TestObject.h)
set_property(TARGET modern-cpp PROPERTY CXX_STANDARD 17)

add_executable(modern-thread thread_main.cpp)
set_property(TARGET modern-thread PROPERTY CXX_STANDARD 17)
target_link_libraries( modern-thread pthread )

add_executable(cpp17 cpp17.cpp)
set_property(TARGET cpp17 PROPERTY CXX_STANDARD 17)

add_executable(move_forward move_forward.cpp)
set_property(TARGET move_forward PROPERTY CXX_STANDARD 17)

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

add_executable(thread_pool_executor_example thread_pool_executor_example.cpp thread_pool_executor.hpp concurrent_queue.hpp)
set_property(TARGET thread_pool_executor_example PROPERTY CXX_STANDARD 11)
target_link_libraries( thread_pool_executor_example pthread )

add_executable(rvalue_reference rvalue_reference.cpp)
set_property(TARGET rvalue_reference PROPERTY CXX_STANDARD 14)
