#include <assert.h>
#include <cmath>     // std::lerp
#include <numeric>   // std::midpoint
#include <iostream>
#include <utility>
#include <array>
#include <memory>
#include <functional>


void to_array() {
  auto arr1 = std::to_array("C-String Literal");
  static_assert(arr1.size() == 17);

  auto arr2 = std::to_array({ 0, 2, 1, 3 });
  static_assert(std::is_same<decltype(arr2), std::array<int, 4>>::value);
  // simple version
  static_assert(std::is_same_v<decltype(arr2), std::array<int, 4>>);

  auto arr3 = std::to_array<long>({ 0, 1, 3 });      // (3)
  static_assert(std::is_same<decltype(arr3), std::array<long, 3>>::value);

  auto arr4 = std::to_array<std::pair<int, float>>( { { 3, .0f }, { 4, .1f }, { 4, .1e23f } });
  static_assert(arr4.size() == 3);                  // (4)
  static_assert(std::is_same<decltype(arr4), std::array<std::pair<int, float>, 3>>::value);
}

template <typename PrefixType>
void startsWith(const std::string& str, PrefixType prefix) {
  std::cout << "starts with " << prefix << ": " << str.starts_with(prefix) << '\n';    // (1)
}

template <typename SuffixType>
void endsWith(const std::string& str, SuffixType suffix) {
  std::cout << "ends with " << suffix << ": " << str.ends_with(suffix) << '\n';
}

void starts_ends_with() {
  std::string helloWorld("Hello World");
  startsWith(helloWorld, helloWorld);                 // (2)
  startsWith(helloWorld, std::string_view("Hello"));  // (3)
  startsWith(helloWorld, 'H');                        // (4)

  endsWith(helloWorld, helloWorld);
  endsWith(helloWorld, std::string_view("World"));
  endsWith(helloWorld, 'd');
}

int plusFunction(int a, int b) {
  return a + b;
}
auto plusLambda = [](int a, int b) {
  return a + b;
};

void bind_front() {
  std::cout << std::endl;

  auto twoThousandPlus1 = std::bind_front(plusFunction, 2000);         // (1)
  std::cout << "twoThousandPlus1(20): " << twoThousandPlus1(20) << std::endl;
  auto twoThousandPlus2 = std::bind_front(plusLambda, 2000);           // (2)
  std::cout << "twoThousandPlus2(20): " << twoThousandPlus2(20) << std::endl;
  auto twoThousandPlus3 = std::bind_front(std::plus<int>(), 2000);     // (3)
  std::cout << "twoThousandPlus3(20): " << twoThousandPlus3(20) << std::endl;

  using namespace std::placeholders;
  auto twoThousandPlus4 = std::bind(plusFunction, 2000, _1);           // (4)
  std::cout << "twoThousandPlus4(20): " << twoThousandPlus4(20) << std::endl;

  auto twoThousandPlus5 =  [](int b) { return plusLambda(2000, b); };  // (5)
  std::cout << "twoThousandPlus5(20): " << twoThousandPlus5(20) << std::endl;
}


int main() {
  // midpoint
  assert( 15 == std::midpoint(10, 20));
  // round down
  assert( 14 == std::midpoint(10, 19));

  // linear interpolation
  assert( 10 == std::lerp(10, 20, 0.0));
  assert( 12 == std::lerp(10, 20, 0.2));
  assert( 21 == std::lerp(10, 20, 1.1));


  to_array();
  starts_ends_with();

  // curring?
  bind_front();
}