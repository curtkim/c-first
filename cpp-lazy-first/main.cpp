#include <assert.h>
#include "Lz/Join.hpp"
#include <iostream>


int main() {
  const std::vector<std::string> strings = {"hello", "world"};
  const auto join = lz::join(strings, ", ");
  std::cout << join << '\n';

  for (const std::string& s : join) {
    std::cout << s << "\n";
  }

  const std::vector<int> ints = {1, 2, 3};
  const auto intJoin = lz::join(ints, ", ");
  std::cout << intJoin << '\n';

  //assert(lz::join(strings, ", ").compare("hello, world") == 0);
}