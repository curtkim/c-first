#include <iostream>
#include <tbb/tbb.h>

void fig_3_5() {
  // step 1: construct the graph
  tbb::flow::graph g;

  // step 2: make the nodes
  tbb::flow::function_node<int, std::string> my_node{
    g,
    tbb::flow::unlimited,
    []( const int& in ) -> std::string {
       std::cout << "received: " << in << std::endl;
       return std::to_string(in);
    }
  };

  tbb::flow::function_node<int, double> my_other_node{
    g,
    tbb::flow::unlimited,
    [](const int& in) -> double {
      std::cout << "other received: " << in << std::endl;
      return double(in);
    }
  };

  // 들어오는 대로 매칭하는 것 같다.
  tbb::flow::join_node<std::tuple<std::string, double>, tbb::flow::queueing> my_join_node{g};

  tbb::flow::function_node<std::tuple<std::string, double>,int> my_final_node{
    g,
    tbb::flow::unlimited,
    [](const std::tuple<std::string, double>& in) -> int {
      std::cout << "final: " << std::get<0>(in) << " and " << std::get<1>(in) << std::endl;
      return 0;
    }
  };

  // step 3: add the edges
  make_edge(my_node, tbb::flow::input_port<0>(my_join_node));
  make_edge(my_other_node, tbb::flow::input_port<1>(my_join_node));
  make_edge(my_join_node, my_final_node);

  // step 4: send messages
  my_node.try_put(1);
  my_other_node.try_put(2);

  my_node.try_put(3);
  my_other_node.try_put(4);

  // step 5: wait for the graph to complete
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
    fig_3_5();
    parallel_time = (tbb::tick_count::now() - t0).seconds();
  }

  std::cout << "parallel_time == " << parallel_time << " seconds" << std::endl;
  return 0;
}