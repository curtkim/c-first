#include "common.h"
#include <vector>

int main()
{
  std::vector<int> vec = {1,2,3};
  nonstd::ring_span<int> span( vec.data(), vec.data() + vec.size(), vec.data(), vec.size());

  std::cout << vec << "\n";
  std::cout << span << "\n";

  // 1. push_back
  span.push_back(4);
  std::cout << vec << "\n";
  std::cout << span << "\n";

  // 2. push_back
  span.push_back(5);
  std::cout << vec << "\n";
  std::cout << span << "\n";

}