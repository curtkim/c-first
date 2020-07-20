#include "tbb/flow_graph.h"
using namespace tbb::flow;

int main() {
  graph g;

  queue_node<int> first_queue(g);
  queue_node<int> second_queue(g);
  split_node< tbb::flow::tuple<int,int> > my_split_node(g);

  output_port<0>(my_split_node).register_successor(first_queue);
  make_edge(output_port<1>(my_split_node), second_queue);

  for(int i = 0; i < 1000; ++i) {
    tuple<int, int> my_tuple(2*i, 2*i+1);
    my_split_node.try_put(my_tuple);
  }
  g.wait_for_all();
}