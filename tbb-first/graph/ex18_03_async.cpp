#include <iostream>
#include <thread>
#include <tbb/flow_graph.h>
#include <tbb/tick_count.h>


class AsyncActivity {
public:
  ~AsyncActivity() {
    asyncThread.join();
  }

  using node_t = tbb::flow::async_node<int, int>;
  using gateway_t = node_t::gateway_type;

  void run(int input, gateway_t &gateway) {
    gateway.reserve_wait();
    asyncThread = std::thread{
      [&, input]() {
        std::cout << std::this_thread::get_id() << " World! Input: " << input << '\n';
        int output = input + 1;
        gateway.try_put(output);
        gateway.release_wait();
      }
    };
  }

private:
  std::thread asyncThread;
};

void async_world() {
  tbb::flow::graph g;
  bool n = false;

  //Source node:
  tbb::flow::source_node<int> in_node{
    g,
    [&](int &a) {
      if (n) return false;
      std::cout << std::this_thread::get_id() << " Async ";
      a = 10;
      n = true;
      return true;
    },
    false
  };

  //Async node:
  AsyncActivity asyncAct;
  using activity_node_t = tbb::flow::async_node<int, int>;
  using gateway_t = activity_node_t::gateway_type;
  activity_node_t a_node{
    g, tbb::flow::unlimited,
    [&asyncAct](int const &input, gateway_t &gateway) {
      asyncAct.run(input, gateway);
    }
  };

  //Output node:
  tbb::flow::function_node<int> out_node{
    g, tbb::flow::unlimited,
    [](int const &a_num) {
      std::cout << std::this_thread::get_id() << " Bye! Received: " << a_num << '\n';
    }
  };

  //Edges:
  make_edge(in_node, a_node);
  make_edge(a_node, out_node);

  //Run!
  in_node.activate();
  g.wait_for_all();
}

int main() {
  std::cout << std::this_thread::get_id() << " main thread" << std::endl;

  tbb::tick_count mainStartTime = tbb::tick_count::now();
  async_world();
  auto time = (tbb::tick_count::now() - mainStartTime).seconds();
  std::cout << "Execution time = " << time << " seconds." << '\n';
  return 0;
}