#include <boost/ut.hpp> // single header
#include <assert.h>

int main() {
  using namespace boost::ut;
  using namespace boost::ut::spec;

  "exceptions/aborts"_test = [] {
    expect(throws<std::runtime_error>([] { throw std::runtime_error{""}; }))
      << "throws runtime_error";
    expect(throws([] { throw 0; })) << "throws any exception";
    expect(nothrow([]{})) << "doesn't throw";
    expect(aborts([] { assert(false); }));
  };

}