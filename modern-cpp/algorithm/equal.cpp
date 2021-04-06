#include <algorithm>
#include <vector>
#include <assert.h>
#include <iostream>
#include <ranges>

int main() {
  std::vector<int> a {0,1,2};
  std::vector<int> b {0,1,2,3,4};

  assert(equal(begin(b), begin(b)+2, begin(a)));

  // c++20을 필요로 한다.
  std::vector<int> range1 {2,3,4,5,6};
  std::vector<int> range2 {2,3,4,5,6};
  assert(std::ranges::equal(range1, range2));
}
