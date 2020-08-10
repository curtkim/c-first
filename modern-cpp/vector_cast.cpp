#include <iostream>
#include <memory>

#include <vector>
#include <algorithm>

using namespace std;

int main() {
  vector<unsigned char> v1 = { 1, 2, 3, 4, 5, 6};

  using Cell = std::array<unsigned char, 3>;

  // compile error
  vector<Cell>* p2 = *reinterpret_cast<vector<Cell>*>(&v1);

}