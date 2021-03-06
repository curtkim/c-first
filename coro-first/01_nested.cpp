#include "generator.h"
#include <iostream>

using coro_exp::generator;

static const double demo_ceiling = 10E5;

#pragma clang diagnostic push

generator<double> doTwice(double v) {
  co_yield v;
  co_yield v;
}

generator<double> fibonacci(const double ceiling) {
  double j = 0;
  double i = 1;
  //co_yield j;
  auto iter = doTwice(j);
  while(iter.next()){
    co_yield iter.getValue();
  }

  if (ceiling > j) {
    do {
      //co_yield i;
      auto iter = doTwice(i);
      while(iter.next()){
        co_yield iter.getValue();
      }

      double tmp = i;
      i += j;
      j = tmp;
    } while (i <= ceiling);
  }
}
#pragma clang diagnostic pop

int main() {
  std::cout << "Example program using C++20 coroutine to implement a Fibonacci Sequence generator" << '\n';
  auto iter = fibonacci(demo_ceiling);
  while(iter.next()) {
    const auto value = iter.getValue();
    std::cout << value << '\n';
  }
}