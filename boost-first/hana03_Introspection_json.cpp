#include <boost/hana.hpp>
#include <string>
#include <assert.h>
#include <iostream>

namespace hana = boost::hana;

struct Car {
  BOOST_HANA_DEFINE_STRUCT(
    Car,
    (std::string, brand),
    (std::string, model)
  );
};
struct Person {
  BOOST_HANA_DEFINE_STRUCT(
    Person,
    (std::string, name),
    (std::string, last_name),
    (int, age)
  );
};

template <typename Xs>
std::string join(Xs&& xs, std::string sep) {
  return hana::fold(hana::intersperse(std::forward<Xs>(xs), sep), "", hana::_ + hana::_);
}
std::string quote(std::string s) { return "\"" + s + "\""; }
template <typename T>
auto to_json(T const& x) -> decltype(std::to_string(x)) {
  return std::to_string(x);
}
std::string to_json(char c) { return quote({c}); }
std::string to_json(std::string s) { return quote(s); }


template <typename Xs>
std::enable_if_t<hana::Sequence<Xs>::value, std::string> to_json(Xs const& xs) {
  auto json = hana::transform(xs, [](auto const& x) {
    return to_json(x);
  });
  return "[" + join(std::move(json), ", ") + "]";
}

template <typename T>
std::enable_if_t<hana::Struct<T>::value, std::string> to_json(T const& x) {
  auto json = hana::transform(hana::keys(x), [&](auto name) {
    auto const& member = hana::at_key(x, name);
    return quote(hana::to<char const*>(name)) + " : " + to_json(member);
  });
  return "{" + join(std::move(json), ", ") + "}";
}

int main() {
  Car bmw{"BMW", "Z3"}, audi{"Audi", "A4"};
  Person john{"John", "Doe", 30};
  auto tuple = hana::make_tuple(john, audi, bmw);
  std::cout << to_json(tuple) << std::endl;
}