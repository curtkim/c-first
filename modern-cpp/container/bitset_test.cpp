#include <bitset>
#include <cassert>
#include <iostream>

int main() {

  constexpr std::bitset<4> b1;
  constexpr std::bitset<4> b2{0xA};
  std::bitset<4> b3{"0011"};
  std::bitset<8> b4{"ABBA", /*length*/ 4, /*0:*/'A', /*1:*/ 'B'};

  std::cout << b1 << "\n";
  std::cout << b2 << "\n";
  std::cout << b3 << "\n";
  std::cout << b4 << "\n";

  b3 |= 0b0100;
  assert(b3 == 0b0111);

  assert(b3.count() == 3);
  return 0;
}
