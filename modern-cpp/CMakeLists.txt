cmake_minimum_required (VERSION 3.2 FATAL_ERROR)
project (modern-cpp)

set (CMAKE_CXX_STANDARD_REQUIRED ON)

### 11

foreach(APP array move_semantic debug_release test_list)
    add_executable(${APP} ${APP}.cpp)
    set_property(TARGET ${APP} PROPERTY CXX_STANDARD 11)
endforeach()

add_executable(thread_pool_executor_example thread_pool_executor_example.cpp thread_pool_executor.hpp concurrent_queue.hpp)
set_property(TARGET thread_pool_executor_example PROPERTY CXX_STANDARD 11)
target_link_libraries( thread_pool_executor_example pthread )


### 14

foreach(APP test_shared_ptr generic rvalue_reference unique cast)
    add_executable(${APP} ${APP}.cpp)
    set_property(TARGET ${APP} PROPERTY CXX_STANDARD 14)
endforeach()

add_executable(test_shared_ptr2 test_shared_ptr2.cpp)
set_property(TARGET test_shared_ptr2 PROPERTY CXX_STANDARD 14)
target_link_libraries( test_shared_ptr2 pthread )

add_executable(Threadpool_test Threadpool_test.cpp Threadpool.hpp)
set_property(TARGET Threadpool_test PROPERTY CXX_STANDARD 14)
target_link_libraries( Threadpool_test pthread )


### 17

foreach(APP
        struct_test
        vector_test
        allocator_test
        main
        cpp17
        move_forward
        move_forward2
        test_string_view_remove_prefix_subfix
        test_string_view_small
        test_string_view_long
        return_value_optimization
        thread_main
        test_async
        test_async2
        variant
        variant_polymorphism
        decltype_first
        const
        promise
        clock_steady
        chrono_timepoint
        chrono_duration
        chrono_to_time_t
        utf8
        producer_consumer
        function
        ref
        question01
        condition_variable_test
        crtp_mixin
        crtp_static_polymorphism
        tag_invoke_test
        )
    message("${APP}")
    add_executable(${APP} ${APP}.cpp)
    set_property(TARGET ${APP} PROPERTY CXX_STANDARD 17)
    target_link_libraries( ${APP} pthread )
endforeach()


foreach(APP
        span_test)
    add_executable(${APP} ${APP}.cpp)
    target_compile_features(${APP} PUBLIC cxx_std_20)
    target_link_libraries( ${APP} pthread )
endforeach()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    add_executable(parallel_transform parallel_transform.cpp)
    set_property(TARGET parallel_transform PROPERTY CXX_STANDARD 17)
    target_link_libraries(parallel_transform PUBLIC OpenMP::OpenMP_CXX)
endif()
