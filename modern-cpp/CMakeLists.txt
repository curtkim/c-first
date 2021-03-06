cmake_minimum_required (VERSION 3.2 FATAL_ERROR)
project (modern-cpp)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

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
foreach(APP producer_consumer test_async test_async2 thread_main condition_variable_test condition_variable_test2 async_pipeline)
    add_executable(${APP} thread_async/${APP}.cpp)
    target_compile_features(${APP} PUBLIC cxx_std_17)
    target_link_libraries(${APP} PUBLIC pthread)
endforeach()

### c++17
foreach(APP cpp17 from_chars set_extract map_try_emplace map_insert_or_assign sampling math_fun array_shared_ptr)
    add_executable(${APP} cpp17/${APP}.cpp)
    target_compile_features(${APP} PUBLIC cxx_std_17)
endforeach()

### c++20
foreach(APP utility stop_token_test)
    add_executable(${APP} cpp20/${APP}.cpp)
    target_compile_features(${APP} PUBLIC cxx_std_20)
    target_link_libraries(${APP} PUBLIC pthread)
endforeach()



add_executable(thread_pool_executor_example thread_async/thread_pool_executor_example.cpp thread_async/thread_pool_executor.hpp thread_async/concurrent_queue.hpp)
set_property(TARGET thread_pool_executor_example PROPERTY CXX_STANDARD 11)
target_link_libraries( thread_pool_executor_example pthread )

add_executable(Threadpool_test thread_async/Threadpool_test.cpp thread_async/Threadpool.hpp)
set_property(TARGET Threadpool_test PROPERTY CXX_STANDARD 14)
target_link_libraries( Threadpool_test pthread )


### memory
foreach(APP move_forward move_forward2 move_semantic exchange test_shared_ptr test_shared_ptr2 rvalue_reference unique ref
        nightmare_move_semantic_for_trivial_class)
    add_executable(${APP} memory/${APP}.cpp)
    target_compile_features(${APP} PUBLIC cxx_std_17)
    target_link_libraries( ${APP} PUBLIC ${CONAN_LIBS} pthread)
endforeach()

### meta
foreach(APP crtp_mixin crtp_static_polymorphism tag_invoke_test static_assert_test constexpr_test constexpr_test2
        type_trait1 type_trait2_decltype type_trait3_enable_if type_trait4_introspect
        if_constexpr_1)
    add_executable(${APP} meta/${APP}.cpp)
    target_compile_features(${APP} PUBLIC cxx_std_17)
    target_link_libraries( ${APP} PUBLIC pthread )
endforeach()

### allocator
foreach(APP allocator_test temporary_allocations)
    add_executable(${APP} allocator/${APP}.cpp)
    target_compile_features(${APP} PUBLIC cxx_std_17)
endforeach()

foreach(APP
        debug_release
        generic
        cast
        struct_test
        main
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
        attribute_pure
        lambda
        lambda2
        reflection_test
        vector_2d
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

foreach(APP type_erasure_with_template type_erasure_with_oop type_erasure_with_sfinae)
    add_executable(${APP} type_erasure/${APP}.cpp)
    target_compile_features(${APP} PUBLIC cxx_std_20)
endforeach()

