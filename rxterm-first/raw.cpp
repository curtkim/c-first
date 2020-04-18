#include <cstdio>
#include <iostream>

int main() {
  printf("%c[%dmHELLO!\n", 0x1B, 32);
  //std::cout << "\e[31m" << "Hello" << "\e[0m" << "World" << std::endl;
  return 0;
}
