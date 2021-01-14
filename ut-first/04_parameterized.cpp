#include <boost/ut.hpp> // single header

int main() {
  using namespace boost::ut;
  using namespace boost::ut::spec;

  for (auto i : std::vector{1, 2, 3}) {
    test("parameterized " + std::to_string(i)) = [i] { // 3 tests
      expect(that % i > 0); // 3 asserts
    };
  }

  "args"_test = [](const auto& arg) {
    expect(arg > 0_i) << "all values greater than 0";
  } | std::vector{1, 2, 3};
}