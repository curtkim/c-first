#include <iostream>
#include <thread>
#include <tbb/flow_graph.h>

// queue_node에 하나는 edge를 만들고, 하나는 try_put으로 merge 효과를
int main() {
  std::cout << std::this_thread::get_id() << " main thread" << std::endl;

  const int DATA_LIMIT = 5;
  int count = 0;

  tbb::flow::graph g;

  tbb::flow::source_node<int> data_generator(
    g,
    [&](int &v) -> bool {
      if (count < DATA_LIMIT) {
        ++count;
        v = count;
        //std::cout << v << std::endl;
        return true;
      } else {
        return false;
      }
    }, false);

  tbb::flow::queue_node<int> queue_node(g);


  tbb::flow::function_node<int, int> process(
    g, tbb::flow::unlimited,
    [&](const int& in) -> int{
      std::cout << std::this_thread::get_id() << " data : " << in << std::endl;
      return 0;
    });


  tbb::flow::make_edge(data_generator, queue_node);
  tbb::flow::make_edge(queue_node, process);

  //buffer_node.try_put(10);
  data_generator.activate();
  queue_node.try_put(10);
  std::cout << "sleep" << std::endl;
  std::this_thread::sleep_for(std::chrono::microseconds(1000));
  queue_node.try_put(11);

  g.wait_for_all();

  return 0;
}