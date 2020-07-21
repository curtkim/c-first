#include "tbb/flow_graph.h"
#include <chrono>
#include <thread>
#include <iostream>
#include <thread>

int main() {
  const int data_limit = 5;
  int count = 0;

  tbb::flow::graph g;

  tbb::flow::function_node<int, int> data_set_preparation(
    g, tbb::flow::unlimited,
    [](int data) {
      printf("Prepare large data set and keep it inside node storage %d\n", data);
      return data;
    });

  tbb::flow::overwrite_node<int> overwrite_storage(g);

  tbb::flow::source_node<int> data_generator(
    g,
    [&](int &v) -> bool {
      if (count < data_limit) {
        ++count;
        v = count;
        //std::cout << v << std::endl;
        return true;
      } else {
        return false;
      }
    }, false);

  tbb::flow::function_node<std::tuple<int, int>, int> process(
    g, tbb::flow::unlimited,
    [&](const std::tuple<int, int>& in) -> int{
      printf("Data to process: %d %d\n", std::get<0>(in), std::get<1>(in));
      return 0;
    });

  // reserving 이면
  // 10 1, 10 2, 10 3, 10 4, 10 5
  // queueing 이면
  // 10 1만 출력
  tbb::flow::join_node<std::tuple<int, int>, tbb::flow::reserving> my_join_node{g};


  tbb::flow::make_edge(data_set_preparation, overwrite_storage);
  tbb::flow::make_edge(overwrite_storage, tbb::flow::input_port<0>(my_join_node));
  tbb::flow::make_edge(data_generator, tbb::flow::input_port<1>(my_join_node));
  tbb::flow::make_edge(my_join_node, process);

  data_set_preparation.try_put(10);
  std::this_thread::sleep_for(std::chrono::microseconds(10));
  data_generator.activate();

  g.wait_for_all();

  return 0;
}