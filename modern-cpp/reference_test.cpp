#include <assert.h>
#include <string>
#include <tuple>

struct Data {
  long a;
  long b;
  long c;
  long d;
};

int main() {
  Data data1 = {1,1,1,1};
  Data data2 = {2,2,2,2};

  Data& ref = data1;
  assert(1 == ref.a);

  ref = data2;
  assert(2 == ref.a);

  std::string a = "1234567890123456789012345678901234567890";
  std::string& b = a;
  std::string c = a; // copy

  assert(&a == &b);
  assert(a.data() == b.data());

  assert(&a != &c);
  assert(a.data() != c.data());

  auto [aa, bb] = std::make_tuple(a, 123);
  assert(a.data() != aa.data());

  const auto& [aaa, bbb] = std::make_tuple(a, 123);
  // 같게 할수 없는 건가?
  assert(a.data() != aaa.data());

}