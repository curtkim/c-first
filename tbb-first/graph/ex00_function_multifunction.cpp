#include <iostream>
#include "tbb/flow_graph.h"

using namespace tbb::flow;
typedef multifunction_node<int, std::tuple<int,int> > buffer4_node;

int main() {
  graph g;

  queue_node<int> even_queue(g);
  queue_node<int> odd_queue(g);

  buffer4_node m_node(g, unlimited, [](const int &i, buffer4_node::output_ports_type &out){
    if(i % 2)
      std::get<1>(out).try_put(i); // put to odd queue
    else
      std::get<0>(out).try_put(i); // put to even queue
  });

  make_edge(output_port<0>(m_node), even_queue);
  make_edge(output_port<1>(m_node), odd_queue);

  for(int i = 0; i < 10; ++i) {
    m_node.try_put(i);
  }
  g.wait_for_all();

  int v;
  bool success = true;
  while(success){
    success = even_queue.try_get(v);
    std::cout << v << " " << success << std::endl;
  }
  std::cout << "===" << std::endl;

  success = true;
  while(success){
    success = odd_queue.try_get(v);
    std::cout << v << " " << success << std::endl;
  }
  return 0;
}