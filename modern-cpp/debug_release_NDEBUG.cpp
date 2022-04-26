#include <iostream>

int main(){
  #ifndef NDEBUG
    std::cout << "DEBUG mode" << std::endl;
  #endif

  return 0;
}