#include <boost/ut.hpp> // single header
#include <assert.h>

using namespace boost::ut;

namespace ut = boost::ut;

ut::suite first = [] {
  using namespace ut;

  "exception"_test = [] {
    expect(throws([] { throw 0; })) << "throws any exception";
  };

  "failure"_test = [] { expect(nothrow([] {})); };
};

ut::suite second = [] {
  "my_test"_test = [] {
    expect(1_i == 2);       // UDL syntax
  };
};

int main() {}
