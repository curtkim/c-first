#include <concepts>
#include <iostream>


template<typename T>
concept Equal = requires(T a, T b){
  {a == b} -> std::convertible_to<bool>;
  {a != b} -> std::convertible_to<bool>;
};

bool areEqual(Equal auto a, Equal auto b){
  return a == b;
}

int main(){
  std::cout << areEqual(1, 5) << "\n";
}
