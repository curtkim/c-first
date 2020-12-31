// C++ High Performance / Ch8 Metaprogramming

#include <type_traits>
#include <vector>
#include <array>
#include <iostream>

auto sign_func = [](const auto& v) -> int {
  using ReferenceType = decltype(v);
  using ValueType = std::remove_reference_t<ReferenceType>;
  if (std::is_unsigned_v<ValueType>) {
    return 1;
  }
  return v < 0 ? -1 : 1;
};

template <typename Range>
auto to_vector(const Range& r) {
  using IteratorType = decltype(r.begin());
  using ReferenceType = decltype(*IteratorType());
  using ValueType = std::decay_t<ReferenceType>;
  return std::vector<ValueType>(r.begin(), r.end());
}

int main() {
  unsigned int a = 1;
  sign_func(a);

  std::array<int, 5> arr = {1,2,3,4,5};
  std::vector<int> vec = to_vector(arr);
  for(int it : vec)
    std::cout << it << " ";
  std::cout << std::endl;
}
