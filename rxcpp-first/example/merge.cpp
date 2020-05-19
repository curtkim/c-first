// https://github.com/ReactiveX/RxCpp/blob/master/Rx/v2/examples/cep/main.cpp

#include <iostream>
#include <thread>
#include <rxcpp/rx.hpp>

namespace rx=rxcpp;
namespace rxsub=rxcpp::subjects;
namespace rxu=rxcpp::util;

#include <cctype>

int main()
{
  std::cout << std::this_thread::get_id() << " main thread" << std::endl;

  auto keys = rx::observable<>::create<int>(
      [](rx::subscriber<int> dest){
        for (;;) {
          int key = std::cin.get();
          dest.on_next(key);
        }
      }).publish();

  auto a = keys.filter([](int key){return std::tolower(key) == 'a';});
  auto g = keys.filter([](int key){return std::tolower(key) == 'g';});

  a.merge(g).subscribe([](int key){
    std::cout << std::this_thread::get_id() << " in subscriber " << key << std::endl;
  });

  // run the loop in create
  keys.connect();
  std::cout << std::this_thread::get_id() << " end" << std::endl;

  return 0;
}
