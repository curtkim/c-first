cmake_minimum_required (VERSION 3.2 FATAL_ERROR)
project (modern-cpp)

set (CMAKE_CXX_STANDARD_REQUIRED ON)

### container
foreach(APP array variant variant_polymorphism vector_test test_list)
    add_executable(${APP} container/${APP}.cpp)
    target_compile_features(${APP} PUBLIC cxx_std_17)
endforeach()

### time
foreach(APP chrono_duration chrono_timepoint chrono_to_time_t clock_steady)
    add_executable(${APP} time/${APP}.cpp)
    target_compile_features(${APP} PUBLIC cxx_std_17)
endforeach()

### thread_async
foreach(APP producer_consumer test_async test_async2 thread_main condition_variable_test)
    add_executable(${APP} thread_async/${APP}.cpp)
    target_compile_features(${APP} PUBLIC cxx_std_17)
endforeach()

add_executable(thread_pool_executor_example thread_async/thread_pool_executor_example.cpp thread_async/thread_pool_executor.hpp thread_async/concurrent_queue.hpp)
set_property(TARGET thread_pool_executor_example PROPERTY CXX_STANDARD 11)
target_link_libraries( thread_pool_executor_example pthread )

add_executable(Threadpool_test thread_async/Threadpool_test.cpp thread_async/Threadpool.hpp)
set_property(TARGET Threadpool_test PROPERTY CXX_STANDARD 14)
target_link_libraries( Threadpool_test pthread )


### memory
foreach(APP move_forward move_forward2 move_semantic exchange test_shared_ptr test_shared_ptr2 rvalue_reference unique ref)
    add_executable(${APP} memory/${APP}.cpp)
    target_compile_features(${APP} PUBLIC cxx_std_17)
    target_link_libraries( ${APP} PUBLIC pthread )
endforeach()

### meta
foreach(APP crtp_mixin crtp_static_polymorphism tag_invoke_test)
    add_executable(${APP} meta/${APP}.cpp)
    target_compile_features(${APP} PUBLIC cxx_std_17)
    target_link_libraries( ${APP} PUBLIC pthread )
endforeach()


foreach(APP
        debug_release
        generic
        cast
        struct_test
        allocator_test
        main
        cpp17
        test_string_view_remove_prefix_subfix
        test_string_view_small
        test_string_view_long
        return_value_optimization
        decltype_first
        const
        promise
        utf8
        function
        question01
        constexpr_test
        attribute_pure
        lambda
        lambda2
        reflection_test
        )
    message("${APP}")
    add_executable(${APP} ${APP}.cpp)
    set_property(TARGET ${APP} PROPERTY CXX_STANDARD 17)
    target_link_libraries( ${APP} pthread )
endforeach()


foreach(APP span_test)
    add_executable(${APP} ${APP}.cpp)
    target_compile_features(${APP} PUBLIC cxx_std_20)
    target_link_libraries( ${APP} pthread )
endforeach()

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    add_executable(parallel_transform thread_async/parallel_transform.cpp)
    set_property(TARGET parallel_transform PROPERTY CXX_STANDARD 17)
    target_link_libraries(parallel_transform PUBLIC OpenMP::OpenMP_CXX)
endif()
