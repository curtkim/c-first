#include <iostream>
#include <thread>
#include <tbb/tbb.h>

void fig_3_3() {
  // step 1: construct the graph
  tbb::flow::graph g;

  // step 2: make the nodes
  tbb::flow::function_node<int, std::string> my_first_node{g,tbb::flow::unlimited, []( const int& in ) -> std::string {
     std::cout << std::this_thread::get_id() << " first node received: " << in << std::endl;
     return std::to_string(in);
   }
  };

  tbb::flow::function_node<std::string> my_second_node{g, tbb::flow::unlimited, []( const std::string& in ) {
     std::cout << std::this_thread::get_id() << " second node received: " << in << std::endl;
   }
  };

  // step 3: add edges
  tbb::flow::make_edge(my_first_node, my_second_node);

  // step 4: send messages
  my_first_node.try_put(10);
  my_second_node.try_put("abc");

  // step 5: wait for graph to complete
  g.wait_for_all();
}

static void warmupTBB() {
  tbb::parallel_for(0, tbb::task_scheduler_init::default_num_threads(), [](int) {
    tbb::tick_count t0 = tbb::tick_count::now();
    while ((tbb::tick_count::now() - t0).seconds() < 0.01);
  });
}

int main(int argc, char *argv[]) {
  warmupTBB();
  double parallel_time = 0.0;
  {
    tbb::tick_count t0 = tbb::tick_count::now();
    fig_3_3();
    parallel_time = (tbb::tick_count::now() - t0).seconds();
  }

  std::cout << "parallel_time == " << parallel_time << " seconds" << std::endl;
  return 0;
}