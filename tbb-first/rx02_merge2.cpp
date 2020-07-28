#include <iostream>
#include <thread>
#include <tbb/flow_graph.h>

// queue_node에 source_node를 2개 연결했는데, 처음것을 삭제한다.
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

  tbb::flow::source_node<int> data_generator2(
    g,
    [&](int &v) -> bool {
      if (count < DATA_LIMIT) {
        ++count;
        v = count+10;
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
  tbb::flow::make_edge(data_generator2, queue_node);
  tbb::flow::make_edge(queue_node, process);

  data_generator.activate();
  data_generator2.activate();

  g.wait_for_all();

  return 0;
}