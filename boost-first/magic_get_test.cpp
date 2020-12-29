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

  Person val{"Edgar Allan Poe", 1809};

  std::cout << boost::pfr::get<0>(val)                // No macro!
            << " was born in " << boost::pfr::get<1>(val) << "\n";  // Works with any aggregate initializables!


  MyStruct s{100, 'H', 3.141593};
  std::cout << "my_struct has " << boost::pfr::tuple_size<MyStruct>::value
            << boost::pfr::io(s) << "\n";
}

