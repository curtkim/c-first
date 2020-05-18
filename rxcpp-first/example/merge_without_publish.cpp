#include "rxcpp/rx.hpp"
namespace rx=rxcpp;
namespace rxsub=rxcpp::subjects;
namespace rxu=rxcpp::util;

#include <cctype>
#include <clocale>

// keys를 publish 하지 않으면
// g에 반응하지 않고, a에만 반응한다.
int main()
{
  auto keys = rx::observable<>::create<int>(
      [](rx::subscriber<int> dest){
        for (;;) {
          int key = std::cin.get();
          dest.on_next(key);
        }
      });

  auto a = keys.
      filter([](int key){return std::tolower(key) == 'a';});

  auto g = keys.
      filter([](int key){return std::tolower(key) == 'g';});

  a.merge(g).subscribe([](int key){
    std::cout << key << std::endl;
  });

  return 0;
}
