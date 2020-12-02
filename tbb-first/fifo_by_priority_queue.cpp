#include <iostream>
#include <tuple>
#include <array>
#include <thread>
#include <tbb/tbb.h>


using namespace std;

// 왜 priority_queue_node의 compare가 호출되지 않는지 모르겠다.
int main() {

  tbb::flow::graph g;

  tbb::flow::function_node<tuple<int, int>, int> process(
    g,
    tbb::flow::unlimited,
    [](tuple<int, int> v) -> int {
      auto [value, delay] = v;
      std::cout << std::this_thread::get_id() << " process " << value << std::endl;
      this_thread::sleep_for(chrono::milliseconds(delay));
      std::cout << std::this_thread::get_id() << " process done " << value << std::endl;
      return value;
    }
  );

  tbb::flow::priority_queue_node<int, function<bool(int, int)>> ordering(
    g,
    [](const int &a, const int &b) -> bool {
      std::cout << a << " " << b << std::endl;
      return a < b;
    });

  tbb::flow::function_node<int> write(
    g,
    tbb::flow::serial,
    [](int v) {
      this_thread::sleep_for(chrono::milliseconds(2000));
      std::cout << std::this_thread::get_id() << " " << v << std::endl;
    }
  );

  tbb::flow::make_edge(process, ordering);
  tbb::flow::make_edge(ordering, write);

  array< tuple<int, int>, 5> data {
    make_tuple(0, 300),
    make_tuple(1, 450),
    make_tuple(2, 500),
    make_tuple(3, 250),
    make_tuple(4, 100),
  };

  for(auto v : data){
    process.try_put(v);
    this_thread::sleep_for(chrono::milliseconds(10));
  }

  g.wait_for_all();

  return 0;
}