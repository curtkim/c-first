// https://github.com/ReactiveX/RxCpp/blob/master/Rx/v2/examples/cep/main.cpp

#include "rxcpp/rx.hpp"

namespace rx=rxcpp;
namespace rxsub=rxcpp::subjects;
namespace rxu=rxcpp::util;

#include <cctype>

int main()
{
  auto _keys = rx::observable<>::create<int>(
      [](rx::subscriber<int> dest){
        for (;;) {
          int key = std::cin.get();
          dest.on_next(key);
        }
      });

  auto keys = _keys.publish();

  auto a = keys.
      filter([](int key){return std::tolower(key) == 'a';});

  auto g = keys.
      filter([](int key){return std::tolower(key) == 'g';});

  a.merge(g).subscribe([](int key){
    std::cout << key << std::endl;
  });

  // run the loop in create
  keys.connect();

  return 0;
}
