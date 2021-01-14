#include <boost/ut.hpp> // single header


int main() {
  using namespace boost::ut;

  "[vector]"_test = [] {
    std::vector<int> v(5);

    expect((5_ul == std::size(v)) >> fatal);

    should("resize bigger") = [v] { // or "resize bigger"_test
      mut(v).resize(10);
      expect(10_ul == std::size(v));
    };

    expect((5_ul == std::size(v)) >> fatal);

    should("resize smaller") = [=]() mutable { // or "resize smaller"_test
      v.resize(0);
      expect(0_ul == std::size(v));
    };
  };

}