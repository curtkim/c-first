#include "tbb/flow_graph.h"
#include <chrono>
#include <thread>
#include <iostream>
#include <thread>

int main() {
  tbb::flow::graph g;

  tbb::flow::function_node<int, int> prepare(
    g, tbb::flow::unlimited,
    [](const int &data) {
      std::cout << std::this_thread::get_id() << " Prepare data: " << data << std::endl;
      return data;
    });

  tbb::flow::buffer_node<int> load_balancer(g);

  tbb::flow::function_node<int, tbb::flow::continue_msg, tbb::flow::rejecting> first_worker(
    g, tbb::flow::serial,
    [](const int &data) {
      std::cout << std::this_thread::get_id() << " Process data with first worker: " << data << std::endl;
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    });

  tbb::flow::function_node<int, tbb::flow::continue_msg, tbb::flow::rejecting> second_worker(
    g, tbb::flow::serial, [](const int &data) {
      std::cout << std::this_thread::get_id() << " Process data with second worker: " << data << std::endl;
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    });

  tbb::flow::make_edge(prepare, load_balancer);
  tbb::flow::make_edge(load_balancer, first_worker);
  tbb::flow::make_edge(load_balancer, second_worker);

  for (int i = 0; i < 10; ++i) {
    prepare.try_put(i);
  }

  g.wait_for_all();

  return 0;
}