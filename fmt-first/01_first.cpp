#include <iostream>
#include <fmt/format.h>

int main() {
  // format
  std::cout << fmt::format("The answer is {}.", 42) << std::endl;

  // stderr
  fmt::print(stderr, "System error code = {}\n", errno);

  // stdout
  fmt::print("Don't {}\n", "panic");
  fmt::print("I'd rather be {1} than {0}.", "right", "happy");

  using namespace fmt::literals;
  // {name} 2번 나온다.
  fmt::print("Hello, {name}! The answer is {number}. Goodbye, {name}.\n",
             "name"_a="World",
             "number"_a=42);

  fmt::print("한글은 잘 나오나?\n");
}