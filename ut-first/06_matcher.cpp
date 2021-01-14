#include <boost/ut.hpp> // single header

int main() {
  using namespace boost::ut;
  using namespace boost::ut::spec;


  "matchers"_test = [] {
    constexpr auto is_between = [](auto lhs, auto rhs) {
      return [=](auto value) {
        return that % value >= lhs and that % value <= rhs;
      };
    };

    expect(is_between(1, 100)(42));
    expect(not is_between(1, 100)(0));
  };

}