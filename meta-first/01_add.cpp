#include <iostream>

template <int a, int b>
struct add {
  static constexpr int value = a + b;
};

int main() {
  auto r = add<1,2>::value;

  std::cout << r << std::endl;
  return 0;
}