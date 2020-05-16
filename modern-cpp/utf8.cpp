#include <iostream>

int main() {

  std::string a = "한글";
  std::cout << a << std::endl;
  std::cout << a.size() << std::endl;
  std::cout << a.length() << std::endl;

  a = "abc";
  std::cout << a << std::endl;
  std::cout << a.size() << std::endl;
  std::cout << a.length() << std::endl;

  return 0;
}