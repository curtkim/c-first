#include <iostream>
#include <string>

int FunctionToDebug(int x, int y) {
  int sum = x + y;
  return sum;
}

int main() {
  std::cout << "Hello world.\n";

  int sum = FunctionToDebug(2, 4);
  std::cout << "Sum of 2 + 4 = " << sum << ".\n";

  return 0;
}
