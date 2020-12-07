#include <boost/hana.hpp>
namespace hana = boost::hana;

#include <cassert>
#include <iostream>
#include <string>


// 1. Give introspection capabilities to 'Person'
struct Person {
  BOOST_HANA_DEFINE_STRUCT(
    Person,
      (std::string, name),
      (int, age)
  );
};

// 2. Write a generic serializer (bear with std::ostream for the example)
auto serialize = [](std::ostream& os, auto const& object) {
  hana::for_each(hana::members(object), [&](auto member) {
    os << member << std::endl;
  });
};

int main(){
  // 3. Use it
  Person john{"John", 30};
  serialize(std::cout, john);
}

