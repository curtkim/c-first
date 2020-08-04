#include <ATen/ATen.h>
#include <iostream>

using namespace std;

int main() {
  at::Tensor a = at::ones({2, 2}, at::kInt);
  at::Tensor b = at::randn({2, 2}); // float type
  auto c = a + b.to(at::kInt);

  cout << a << endl;
  cout << b << endl;
  cout << c << endl;

  cout << a.sum(at::kInt) << endl;

  return 0;
}

