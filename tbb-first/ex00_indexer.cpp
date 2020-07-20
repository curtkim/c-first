#include <cstdio>
#include <iostream>
#include <thread>
#include <tbb/flow_graph.h>

using namespace tbb::flow;

int main() {
  graph g;
  function_node<int, int> f1(
    g, unlimited,
    [](const int &i) { return 2 * i; });

  function_node<float, float> f2(
    g, unlimited,
    [](const float &f) { return f / 2; });

  typedef indexer_node<int, float> my_indexer_type;
  my_indexer_type o(g);

  function_node<my_indexer_type::output_type> f3(
    g, unlimited,
    [](const my_indexer_type::output_type &v) {
      std::cout << std::this_thread::get_id() << " " << v.tag() << std::endl;
      if (v.tag() == 0) {
        printf("Received an int %d\n", cast_to<int>(v));
      } else {
        printf("Received a float %f\n", cast_to<float>(v));
      }
    }
  );

  make_edge(f1, input_port<0>(o));
  make_edge(f2, input_port<1>(o));
  make_edge(o, f3);

  f1.try_put(3);
  f2.try_put(3);
  g.wait_for_all();
  return 0;
}