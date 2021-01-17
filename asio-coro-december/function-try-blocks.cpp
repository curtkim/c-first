#include <iostream>

int add(int a , int b)
try {
  return a+b;
}
catch(...){

}

int main() {
  std::cout << add(1,2) << std::endl;
  return 0;
}