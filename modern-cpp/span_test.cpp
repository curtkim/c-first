#include <algorithm>
#include <cstddef>
#include <iostream>
#include <span>

template<class T, std::size_t N> [[nodiscard]]
constexpr auto slide(std::span<T,N> s, std::size_t offset, std::size_t width) {
  return s.subspan(offset, offset + width <= s.size() ? width : 0U);
}

template<class T, std::size_t N, std::size_t M> [[nodiscard]]
constexpr bool starts_with(std::span<T,N> data, std::span<T,M> prefix) {
  return data.size() >= prefix.size()
         && std::equal(prefix.begin(), prefix.end(), data.begin());
}

template<class T, std::size_t N, std::size_t M> [[nodiscard]]
constexpr bool ends_with(std::span<T,N> data, std::span<T,M> suffix) {
  return data.size() >= suffix.size()
         && std::equal(data.end() - suffix.size(), data.end(),
                       suffix.end() - suffix.size());
}

template<class T, std::size_t N, std::size_t M> [[nodiscard]]
constexpr bool contains(std::span<T,N> span, std::span<T,M> sub) {
  return std::search(span.begin(), span.end(), sub.begin(), sub.end())
         != span.end();
}

void print(const auto& seq) {
  for (const auto& elem : seq) std::cout << elem << ' ';
  std::cout << '\n';
}

int main()
{
  constexpr int a[] { 0, 1, 2, 3, 4, 5, 6, 7, 8 };
  constexpr int b[] { 8, 7, 6 };

  for (std::size_t offset{}; ; ++offset) {
    constexpr std::size_t width{6};
    auto s = slide(std::span{a}, offset, width);
    if (s.empty())
      break;
    print(s);
  }

  static_assert(starts_with(std::span{a}, std::span{a,4})
                && starts_with(std::span{a+1, 4}, std::span{a+1,3})
                && !starts_with(std::span{a}, std::span{b})
                && !starts_with(std::span{a,8}, std::span{a+1,3})
                && ends_with(std::span{a}, std::span{a+6,3})
                && !ends_with(std::span{a}, std::span{a+6,2})
                && contains(std::span{a}, std::span{a+1,4})
                && !contains(std::span{a,8}, std::span{a,9}));
}