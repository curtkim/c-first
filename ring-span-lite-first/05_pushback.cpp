#include "common.h"
#include <vector>
#include <array>
#include <assert.h>

int main()
{
  {
    std::cout << "vector\n";
    std::vector<int> vec = {1, 2, 3};
    nonstd::ring_span<int> span(vec.data(), vec.data() + vec.size(), vec.data(), vec.size());
    std::cout << "sizeof(nonstd::ring_span<int>) = " << sizeof(span) << "\n";

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
    assert(2 == span[0]);
    assert(3 == span[1]);
    assert(4 == span[2]);

    // 2. push_back
    span.push_back(5);
    assert(3 == span[0]);
    assert(4 == span[1]);
    assert(5 == span[2]);
  }

  {
    std::cout << "empty array\n";
    std::array<int,3> arr = {};
    nonstd::ring_span<int> span(arr.data(), arr.data() + arr.size(), arr.data(), 0);

    assert(span.empty());

    // 1. push_back
    span.push_back(1);
    assert(1 == span[0]);

    span.push_back(2);
    assert(2 == span.size());
    assert(1 == span[0]);
    assert(2 == span[1]);
  }


}