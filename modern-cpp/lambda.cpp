#include <iostream>
#include <vector>
#include <functional>

using namespace std;


void test_lambda() {
  [] { std::cout << "Hello World!" << std::endl; } ();

  auto func = [] { std::cout << "Hello World!" << std::endl; };
  func();

  auto func2 = [] ( int n ) { std::cout << "I have " << n << " girl friends" << std::endl; };
  func2 ( 3 );
}

void test_lambda_return() {
  auto func1 = [] { return 3.14; };
  auto func2 = [] (float f) { return f; };
  auto func3 = [] () -> float { return 3.14; };

  float f1 = func1();
  float f2 = func2( 3.14f );
  float f3 = func3();
}

void test_lambda_each() {
  std::vector<int> v1;
  v1.emplace_back( 10 );
  v1.emplace_back( 20 );
  v1.emplace_back( 30 );

  std::for_each ( v1.begin(), v1.end(),
                  [] ( int n ) { std::cout << n << std::endl; }
  );
}

std::function < void() > funcReturnLambda() {
  std::string str("This is Lambda!");
  return [=] { std::cout << "What!?" << str << std::endl; };
}

void test_lambda_capture_reference() {
  int x = 100;
  [&]() {
    std::cout << x << std::endl;
    x = 200;
  } ();
  std::cout << x << std::endl; // print x = 200
}

void test_lambda_capture_copy() {
  int x = 100;
  [=]() { std::cout << x << std::endl; } (); // print x = 100
  [=]() mutable {
    std::cout << x << std::endl;
    x = 200;
  }(); // x = 200
  std::cout << x << std::endl; // print x = 100
}

void test_lambda_capture3(){
  int sum = 0;
  int divisor = 3;
  vector<int> numbers { 1, 2, 3, 4, 5, 10, 15, 20, 25, 35, 45, 50 };
  for_each(numbers.begin(), numbers.end(), [divisor, &sum] (int y)
  {
    if (y % divisor == 0)
    {
      cout << y << endl;
      sum += y;
    }
  });

  cout << sum << endl;
}

void test_generic_lambda() {
  // gerneric lambda
  auto sum = [](auto a, decltype(a) b) { return a + b; };

  int i = sum( 3, 4 );
  double d = sum ( 3.14, 2.77 );

  cout << i << " " << d << endl;
}

int main(){
  test_lambda();
  test_lambda_return();
  test_lambda_each();

  auto func = funcReturnLambda();
  func();
  funcReturnLambda()();

  test_lambda_capture_reference();
  test_lambda_capture_copy();
  test_lambda_capture3();
  test_generic_lambda();

}