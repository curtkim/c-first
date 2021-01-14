#include <boost/ut.hpp> // single header
// import boost.ut;        // single module (C++20)

int main() {
  using namespace boost::ut;

  expect(1_i == 2);       // UDL syntax
  expect(1 == 2_i);       // UDL syntax
  expect(that % 1 == 2);  // Matcher syntax
  expect(eq(1, 2));       // eq/neq/gt/ge/lt/le

  expect(42l == 42_l and 1 == 2_i); // compound expression
  expect(42l == 42_l and 1 == 2_i) << "additional info";

  "hello world"_test = [] {
    int i = 43;
    expect(42_i == i);
  };

}