#include <iostream>
#include <fmt/format.h>

void basic() {
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

void print_with_argument_id() {
  std::cout << '\n';
  std::cout << fmt::format("{} {}: {}!\n", "Hello", "World", 2020);
  std::cout << fmt::format("{1} {0}: {2}!\n", "World", "Hello", 2020);
  std::cout << fmt::format("{0} {0} {1}: {2}!\n", "Hello", "World", 2020);
  std::cout << fmt::format("{0}: {2}!\n", "Hello", "World", 2020);
  std::cout << '\n';
}

void fill_align() {
  std::cout << '\n';
  int num = 2020;
  std::cout << fmt::format("{:6}", num) << '\n';
  std::cout << fmt::format("{:6}", 'x') << '\n';
  std::cout << fmt::format("{:*<6}", 'x') << '\n';
  std::cout << fmt::format("{:*>6}", 'x') << '\n';
  std::cout << fmt::format("{:*^6}", 'x') << '\n';
  std::cout << fmt::format("{:6d}", num) << '\n';
  std::cout << fmt::format("{:6}", true) << '\n';
  std::cout << '\n';
}

void align(){
  std::string word = "hello";

  std::cout << fmt::format("{:=<20}", word) << "\n";
  std::cout << fmt::format("{:->20}", word) << "\n";
  std::cout << fmt::format("{:/^20}", word) << "\n";
}

int main() {
  print_with_argument_id();
  fill_align();
  align();
}