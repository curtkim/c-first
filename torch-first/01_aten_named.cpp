#include <ATen/ATen.h>
#include <iostream>

using namespace std;

int main() {
  /*
  const at::ArrayRef NCHW{
    at::Dimname::fromSymbol(c10::Symbol::dimname("N")), // batch
    at::Dimname::fromSymbol(c10::Symbol::dimname("C")), // channel
    at::Dimname::fromSymbol(c10::Symbol::dimname("H")), // height
    at::Dimname::fromSymbol(c10::Symbol::dimname("W"))  // width
  };
  */

  at::ArrayRef HW{
    at::Dimname::fromSymbol(c10::Symbol::dimname("H")),
    at::Dimname::fromSymbol(c10::Symbol::dimname("W"))
  };

  at::Tensor a = at::ones({2, 2}, HW, at::kInt);
  cout << "sizeof(a)=" << sizeof(a) << endl;          // 8
  cout << "byteof(a) = " << a.nbytes() << endl;       // 16 = 4*4
  cout << a.get_named_tensor_meta()->names() << endl;

  at::Tensor b = at::randn({2, 2}).mul(2);  // float type
  cout << "sizeof(b)=" << sizeof(b) << endl;          // 8
  cout << "byteof(b) = " << b.nbytes() << endl;       // 16 = 4*4

  auto c = a + b.to(at::kInt);
  cout << "sizeof(c)=" << sizeof(c) << endl;          // 8
  cout << "byteof(c)=" << c.nbytes() << endl;         // 16 = 4*4
  cout << "c.has_names(): " << c.has_names() << endl;

  cout << a << endl;
  cout << b << endl;
  cout << c << endl;

  cout << a.sum(at::kInt) << endl;

  return 0;
}

