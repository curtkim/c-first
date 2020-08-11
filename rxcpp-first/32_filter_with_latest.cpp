#include <memory>
#include <thread>
#include <rxcpp/rx.hpp>


namespace Rx {
using namespace rxcpp;
using namespace rxcpp::sources;
using namespace rxcpp::operators;
using namespace rxcpp::util;
}
using namespace Rx;

using namespace std;

#include <chrono>

using namespace std::chrono_literals;


int main() {
  std::cout << "main thread " << std::this_thread::get_id() << endl;

  auto values = rxcpp::sources::range(0, 10);

  auto even = values.filter([](int a){
    return a % 2 == 0;
  });
  auto odd = values.filter([](int a){
    return a % 2 == 1;
  });

  even.with_latest_from(odd)
  .subscribe([](std::tuple<int, int> c){
    auto [a, b] = c;
    std::cout << a << " " << b << std::endl;
  });

  return 0;
}
