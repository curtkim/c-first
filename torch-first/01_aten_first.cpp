#include <ATen/ATen.h>
#include <iostream>

using namespace std;

const c10::ArrayRef RC{
  at::Dimname::fromSymbol(c10::Symbol::dimname("R")),
  at::Dimname::fromSymbol(c10::Symbol::dimname("C"))
};


int main() {
  at::Tensor a = at::ones({2, 2}, RC, at::kInt);
  cout << a.get_named_tensor_meta()->names() << endl;

  at::Tensor b = at::randn({2, 2}); // float type
  auto c = a + b.to(at::kInt);

  cout << "c.has_names(): " << c.has_names() << endl;

  cout << a << endl;
  cout << b << endl;
  cout << c << endl;

  cout << a.sum(at::kInt) << endl;

  return 0;
}

