#include <ATen/ATen.h>
#include <iostream>


int main() {
  at::Tensor a = at::ones({2, 2}, at::kInt);
  at::Tensor b = at::randn({2, 2});
  auto c = a + b.to(at::kInt);

  std::cout << a << std::endl;
  std::cout << b << std::endl;
  std::cout << c << std::endl;

  return 0;
}

