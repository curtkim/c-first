#include <iostream>
#include "mystring.hpp"

int main(){
  std::cout << mystring::substring("abcdefg", 1, 2) << std::endl;
  return 0;
}