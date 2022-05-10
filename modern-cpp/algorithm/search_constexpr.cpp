// https://www.cppstories.com/2018/08/searchers/

#include <algorithm>
#include <iostream>
#include <functional> // searchers

constexpr bool IsPresent(std::string_view pattern, std::string_view str) {
  // only default_searcher is constexpr in cpp20
  auto it = std::search(str.begin(), str.end(),
                        std::default_searcher(pattern.begin(), pattern.end()));
  return it != str.end();
}

int main() {
  static_assert(IsPresent("hello", "super hello world") == true);
  static_assert(IsPresent("HELLO", "super hello world") == false);
}