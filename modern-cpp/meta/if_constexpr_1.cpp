#include <iostream>

struct Bear { auto roar() const { std::cout << "roar"; } };
struct Duck { auto quack() const { std::cout << "quack"; } };

/*
template <typename Animal>
auto speak(const Animal& a) {
  if (std::is_same_v<Animal, Bear>) { a.roar(); }
  else if (std::is_same_v<Animal, Duck>) { a.quack(); }
}
*/

template <typename Animal>
auto speak(const Animal& a) {
  if constexpr (std::is_same_v<Animal, Bear>) { a.roar(); }
  else if constexpr (std::is_same_v<Animal, Duck>) { a.quack(); }
}


int main() {
  auto bear = Bear{};
  speak(bear);
}

