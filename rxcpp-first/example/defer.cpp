#include <iostream>
#include <thread>
#include <rxcpp/rx.hpp>

namespace rx=rxcpp;
namespace rxsub=rxcpp::subjects;
namespace rxu=rxcpp::util;

// 왜 필요한지 모르겠다.
int main()
{
  std::cout << std::this_thread::get_id() << " main thread" << std::endl;

  auto observable_factory = [](){return rxcpp::observable<>::range(1, 3);};

  auto values = rxcpp::observable<>::defer(observable_factory);
  values.
      subscribe(
      [](int v){printf("OnNext: %d\n", v);},
      [](){printf("OnCompleted\n");});

  return 0;
}
