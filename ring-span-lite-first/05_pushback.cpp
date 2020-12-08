#include "common.h"
#include <vector>
#include <array>

int main()
{
  {
    std::cout << "vector\n";
    std::vector<int> vec = {1, 2, 3};
    nonstd::ring_span<int> span(vec.data(), vec.data() + vec.size(), vec.data(), vec.size());

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

  {
    std::cout << "array\n";
    std::array<int,3> arr = {1,2,3};
    nonstd::ring_span<int> span(arr.data(), arr.data() + arr.size(), arr.data(), arr.size());

    std::cout << arr << "\n";
    std::cout << span << "\n";

    // 1. push_back
    span.push_back(4);
    std::cout << arr << "\n";
    std::cout << span << "\n";

    // 2. push_back
    span.push_back(5);
    std::cout << arr << "\n";
    std::cout << span << "\n";
  }
}