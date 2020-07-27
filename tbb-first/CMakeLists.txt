cmake_minimum_required(VERSION 2.8.12)
project(MD5Encrypter)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

file(COPY input1.png DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)
file(COPY input2.png DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/bin)

foreach(APP
        ex00_join_key_match ex00_indexer ex00_split_node ex00_buffer ex00_queue_node
        ex00_overwrite_join_reserving ex00_multifunction
        ex00_single_broadcast
        ex01_04_parallel_invoke ex01_05_for_each_par ex01_07_plain_loop
        ex01_10_flow ex01_11_flow_parallel_for ex01_12_flow_parallel_for_unseq
        ex02_10_parallel_reduce ex02_27_parallel_pipeline ex02_30_parallel_pipeline
        ex03_03 ex03_05_join_by_queueing ex03_06_source_or_try_put ex03_10_join_tag_match
        ex04_04_par_unseq ex04_11_reduce
        ex05_03_histogram ex05_05 ex05_20_atomic ex05_20_atomic_cache_aligned ex05_22_tls_ets ex05_26_conbinable ex05_27_parallel_reduce
        ex06_01_concurrent_map ex06_02_concurrent_queue ex06_10
        ex07_09_parallel_for
        ex10_04_parallel_invoke ex10_06_task_group ex10_08 ex10_11 ex10_11_bypass ex10_15_recycle
        ex17_01 ex17_02 ex17_03 ex17_04
        ex17_09_transpose ex17_10_transpose_tile
        ex17_13_no_limit ex17_15_limit ex17_16_rejecting ex17_17_token_buffer
        ex17_19_arena ex17_21_sequencer ex17_22_order_multifunction ex17_33_composite
        ex18_03_async
        rx01_with_latest rx02_merge rx03_buffer)
    add_executable(${APP} ${APP}.cpp)
    target_link_libraries(${APP} ${CONAN_LIBS})
    target_compile_features(${APP} PUBLIC cxx_std_17)
endforeach()

#add_executable(ex07_09_tbbmalloc ex07_09.cpp)
#target_link_libraries(ex07_09_tbbmalloc ${CONAN_LIBS} tbbmalloc_proxy)
#target_compile_features(ex07_09_tbbmalloc PUBLIC cxx_std_17)
