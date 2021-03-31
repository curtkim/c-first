#include "Lz/GroupBy.hpp"

int main() {
  std::vector<std::string> vec = {
    "hello", "hellp", "i'm", "done"
  };

  auto grouper = lz::groupBy(vec, [](const std::string& s) { return s.length(); });

  for (auto &&[first, second] : grouper) {
    fmt::print("===\n");
    fmt::print("String length group: {}\n", first);
    for (const auto &str : second) {
      fmt::print("{} : {}\n", first, str);
    }
  }
}