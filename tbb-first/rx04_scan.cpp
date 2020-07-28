#include <iostream>
#include <thread>
#include <tbb/flow_graph.h>

// merge
// interval
// with_latest
// window?
// pairwise
// flat_map?
// scan


int main() {
  std::cout << std::this_thread::get_id() << " main thread" << std::endl;

  tbb::flow::graph g;

  int sum = 0;
  tbb::flow::function_node<int, int> sum_node{
    g, tbb::flow::serial,
    [&sum](int in) -> int {
      sum += in;
      return sum;
    }
  };

  tbb::flow::function_node<int, int> print_node{
    g, tbb::flow::serial,
    [](int in) -> int {
      std::cout << std::this_thread::get_id() << " " << in << std::endl;
      return in;
    }
  };

  tbb::flow::make_edge(sum_node, print_node);

  for(int i = 0; i < 10; i++)
    sum_node.try_put(i);

  g.wait_for_all();

  return 0;
}