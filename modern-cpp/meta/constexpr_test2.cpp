#include <iostream>

// C++11 constexpr functions use recursion rather than iteration
// (C++14 constexpr functions may use local variables and loops)
constexpr int factorial(int n)
{
  return n <= 1 ? 1 : (n * factorial(n - 1));
}

constexpr int factorial2(int n)
{
  int result = 1;
  for(int i = 2; i <= n; i++)
    result *= i;
  return result;
}

// output function that requires a compile-time constant, for testing
template<int n>
struct constN
{
  constN() { std::cout << n << '\n'; }
};

int main()
{
  std::cout << "4! = " ;
  constN<factorial(4)> out1; // computed at compile time

  std::cout << "4! = " ;
  constN<factorial2(4)> out2; // computed at compile time

  volatile int k = 8; // disallow optimization using volatile
  std::cout << k << "! = " << factorial(k) << '\n'; // computed at run time
}