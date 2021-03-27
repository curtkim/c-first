#include <assert.h>
#include <iostream>
#include <string>
#include <boost/pfr.hpp> // Precise and Flat Reflection


struct Person {
  std::string name;
  unsigned birth_year;
};

struct MyStruct { // no ostream operator defined!
  int i;
  char c;
  double d;
};


int main() {

  // 2 field
  Person val{"Edgar Allan Poe", 1809};
  assert(boost::pfr::get<0>(val).compare("Edgar Allan Poe") == 0);
  assert(boost::pfr::get<1>(val) == 1809);


  // 3 field
  MyStruct s{100, 'H', 3.141593};
  static_assert( 3 == boost::pfr::tuple_size<MyStruct>::value);

  std::ostringstream oss;
  oss << boost::pfr::io(s);
  assert(oss.str().compare("{100, H, 3.14159}") == 0);
}

