#include "tbb/flow_graph.h"
#include <chrono>
#include <thread>
#include <iostream>
#include <thread>

// An unbounded buffer of messages of type T.
// Messages are forwarded in first-in first-out (FIFO) order.
int main() {
  using namespace tbb::flow;
  graph g;

  function_node<int, int> prepare(
    g, serial,
    [](const int &data) {
      std::cout << std::this_thread::get_id() << " Prepare data: " << data << std::endl;
      return data;
    });

  queue_node<int> load_balancer(g);

  function_node<int, continue_msg, rejecting> first_worker(
    g, serial,
    [](const int &data) {
      std::cout << std::this_thread::get_id() << " Process data with first worker: " << data << std::endl;
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    });

  function_node<int, continue_msg, rejecting> second_worker(
    g, serial, [](const int &data) {
      std::cout << std::this_thread::get_id() << " Process data with second worker: " << data << std::endl;
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    });

  make_edge(prepare, load_balancer);
  make_edge(load_balancer, first_worker);
  make_edge(load_balancer, second_worker);

  for (int i = 0; i < 10; ++i) {
    prepare.try_put(i);
  }

  g.wait_for_all();

  return 0;
}