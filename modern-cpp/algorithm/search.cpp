#include <algorithm>
#include <iostream>
#include <functional> // searchers
#include <iomanip>    // quoted

int main() {
  std::string str = "Hello Super World";
  std::string needle = "Super";

  std::cout << "looking for " << std::quoted(needle) << " in " << std::quoted(str) << '\n';

  auto it = search(str.begin(), str.end(),
                   std::boyer_moore_searcher(needle.begin(), needle.end()));
  if (it != str.end())
    std::cout << "found at pos " << std::distance(str.begin(), it) << '\n';
  else
    std::cout << "...not found\n";
}
