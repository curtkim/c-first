#include "generator.h"
#include <iostream>

using coro_exp::generator;

generator<int> gen(int start) {
  for(int i = 0; ; i++)
    co_yield start+i;
}

generator<int> take_until(generator<int>& source, int count){
  for(int i = 0; i < count; i++) {
    source.next();
    co_yield source.getValue();
  }
}

int main() {
  auto source = gen(10);
  auto iter = take_until(source, 5);
  while(iter.next()) {
    const auto value = iter.getValue();
    std::cout << value << '\n';
  }
}