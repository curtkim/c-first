#include <iostream>
#include <functional>

using namespace std;


template <typename Functor>
void f(Functor functor) {
  cout << __PRETTY_FUNCTION__ << endl;
  functor();
}

/* Or alternatively you can use this
void f(function<int(int)> functor) {
    cout << __PRETTY_FUNCTION__ << endl;
}
*/

int g() {
  static int i = 0;
  return i++;
}

int main() {
  auto lambda_func = [i = 0]() mutable { return i++; };
  f(lambda_func); // Pass lambda
  f(g);            // Pass function
}