#include <torch/script.h> // One-stop header.
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>

#include <iostream>


int main() {
  torch::Tensor a = torch::ones({2, 2}, torch::requires_grad());
  torch::Tensor b = torch::randn({2, 2});
  auto c = a + b;

  std::cout << a << std::endl;
  std::cout << b << std::endl;
  std::cout << c << std::endl;
  //std::cout << a.grad() << std::endl;


  c.backward(); // a.grad() will now hold the gradient of c w.r.t. a
  std::cout << "===" << std::endl;
  std::cout << a.grad() << std::endl;

  return 0;
}
