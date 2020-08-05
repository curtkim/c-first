#include <ATen/ATen.h>
#include <torch/torch.h>

#include <iostream>

using namespace std;

void by_torch() {
  torch::Tensor a = torch::ones({2, 2}, torch::kInt);
  torch::Tensor b = torch::randn({2, 2}); // float type
  auto c = a + b.to(torch::kInt);

  cout << a << endl;
  cout << b << endl;
  cout << c << endl;

  cout << a.sum(torch::kInt) << endl;
}

int main() {
  at::Tensor a = at::ones({2, 2}, at::kInt);
  at::Tensor b = at::randn({2, 2}); // float type
  auto c = a + b.to(at::kInt);

  cout << a << endl;
  cout << b << endl;
  cout << c << endl;

  cout << a.sum(at::kInt) << endl;

  by_torch();
  return 0;
}

