#include <iostream>
#include <chrono>

long fibonacci(unsigned n)
{
  if (n < 2) return n;
  return fibonacci(n-1) + fibonacci(n-2);
}

int main()
{
  auto start = std::chrono::steady_clock::now();
  std::cout << "f(42) = " << fibonacci(42) << '\n';
  auto end = std::chrono::steady_clock::now();

  std::chrono::duration<double> elapsed_seconds = end-start;
  std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
}