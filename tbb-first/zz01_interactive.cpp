#include <iostream>
#include <algorithm>
#include <thread>
#include <tbb/tbb.h>
#include <tbb/flow_graph.h>

int main() {
  std::cout << std::this_thread::get_id() << " main thread" << std::endl;

  tbb::task_scheduler_init init{2};
  tbb::flow::graph g;

  tbb::flow::function_node<std::string,std::string> process(
    g,tbb::flow::unlimited,
    [](const std::string& in)-> std::string{
      std::string out(in);
      std::transform(in.begin(), in.end(), out.begin(), ::toupper);
      return out;
    });

  tbb::flow::function_node<std::string,int> output(
    g,tbb::flow::unlimited,
    [](const std::string& in)-> int{
      std::cout << std::this_thread::get_id() << " " << in << std::endl;
      return 0;
    });

  tbb::flow::make_edge(process, output);

  char x;
  while(std::cin>>x)
  {
    process.try_put(std::string(1, x));
  }

  return 0;
}