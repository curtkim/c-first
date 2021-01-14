#include <boost/ut.hpp> // single header

int main() {
  using namespace boost::ut;
  using namespace boost::ut::spec;

  describe("vector") = [] {
    std::vector<int> v(5);
    expect((5_ul == std::size(v)) >> fatal);

    it("should resize bigger") = [v] {
      mut(v).resize(10);
      expect(10_ul == std::size(v));
    };
  };

}