#include <iostream>
#include <fmt/format.h>

int main() {
  std::cout << fmt::format("The answer is {}.", 42) << std::endl;

  fmt::print(stderr, "System error code = {}\n", errno);

  fmt::print("Don't {}\n", "panic");
  fmt::print("I'd rather be {1} than {0}.", "right", "happy");

  using namespace fmt::literals;
  fmt::print("Hello, {name}! The answer is {number}. Goodbye, {name}.",
             "name"_a="World", "number"_a=42);
}