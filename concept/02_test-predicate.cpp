#include <concepts>
#include <iostream>
#include <functional>

template<std::predicate T>
void f(T){
  if(T())
    std::cout << "success" << std::endl;
  else
    std::cout << "fail" << std::endl;
}

bool even(int v){
  return v % 2 == 0;
}

int main() {
  auto a = []()-> bool {
    return false;
  };
  f(a);
  f([](){return even(1);});

  return 0;
}
