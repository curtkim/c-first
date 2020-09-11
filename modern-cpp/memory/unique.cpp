#include <iostream>
#include <memory>

using namespace std;

// https://shaharmike.com/cpp/unique-ptr/
int main() {
  auto u_int = std::make_unique<int>(123);
  cout << (*u_int == 123) << endl;

  auto u_string = std::make_unique<std::string>(3, '#');
  cout << (*u_string == "###") << endl;
}