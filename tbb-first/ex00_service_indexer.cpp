#include <cstdio>
#include <iostream>
#include <thread>
#include <tbb/flow_graph.h>

int main() {
  using namespace tbb::flow;

  graph g;
  function_node<int, std::string> f1(
    g, unlimited,
    [](const int &i) { return 2 * i; });

  function_node<float, float> f2(
    g, unlimited,
    [](const float &f) { return f / 2; });

  typedef indexer_node<std::string, float> my_indexer_type;
  my_indexer_type indexer(g);

  function_node<my_indexer_type::output_type> f3(
    g, unlimited,
    [](const my_indexer_type::output_type &v) {
      std::cout << std::this_thread::get_id() << " tag=" << v.tag() << std::endl;
      if (v.tag() == 0) {
        printf("Received an int %d\n", cast_to<std::string>(v));
      } else {
        printf("Received a float %f\n", cast_to<float>(v));
      }
    }
  );

  make_edge(f1, input_port<0>(indexer));
  make_edge(f2, input_port<1>(indexer));
  make_edge(indexer, f3);

  f1.try_put(3);
  f2.try_put(3.0);
  g.wait_for_all();
  return 0;
}