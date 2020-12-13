#include <boost/hana.hpp>
#include <string>
#include <assert.h>
#include <iostream>

namespace hana = boost::hana;

// 1. Checking expression validity
auto has_toString = hana::is_valid([](auto&& obj) -> decltype(obj.toString()) { });

template <typename T>
std::string optionalToString(T const& obj) {
  return hana::if_(has_toString(obj),
                   [](auto& x) { return x.toString(); },
                   [](auto& x) { return "toString not defined"; }
  )(obj);
}

struct A{};
struct B{
  std::string toString() {
    return "B";
  }
};
BOOST_HANA_CONSTANT_CHECK(!has_toString(A{}));
BOOST_HANA_CONSTANT_CHECK(has_toString(B{}));

void checking_expression_validity(){
  assert(optionalToString(A{}) == "toString not defined");
  assert(optionalToString(B{}) == "B");
}

// 2.
auto has_member = hana::is_valid([](auto&& x) -> decltype((void)x.member) { });
struct Foo { int member[4]; };
struct Bar { };
BOOST_HANA_CONSTANT_CHECK(has_member(Foo{}));
BOOST_HANA_CONSTANT_CHECK(!has_member(Bar{}));

int main() {
  checking_expression_validity();
}