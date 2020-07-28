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

typedef tbb::flow::multifunction_node<int, std::tuple<std::vector<int>>> buffer4_node;

// buffer(4를 묶어서)
int main() {
  std::cout << std::this_thread::get_id() << " main thread" << std::endl;

  const int DATA_LIMIT = 12;
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

  std::vector<int> temp;
  buffer4_node process(
    g, tbb::flow::unlimited,
    [&temp](const int &in, buffer4_node::output_ports_type &out){
      //std::cout << std::this_thread::get_id() << " data : " << in << std::endl;
      temp.push_back(in);
      if( temp.size() >= 4){
        // TODO 메모리를 효율적으로
        std::vector<int> temp2;
        temp2 = temp;
        std::get<0>(out).try_put(temp2);
        temp.clear();
      }
    });

  tbb::flow::function_node<std::vector<int>,int> output(
    g,
    tbb::flow::unlimited,
    [](const std::vector<int>& in)-> int{
      for(auto i : in)
        std::cout << i << " ";
      std::cout << std::endl;
      return 0;
    });

  tbb::flow::make_edge(data_generator, process);
  tbb::flow::make_edge(process, output);

  data_generator.activate();
  g.wait_for_all();

  return 0;
}