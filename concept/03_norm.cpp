// https://www.youtube.com/watch?v=B_KjoLid5gw&t=166s
#include <iostream>
#include <vector>
#include <cmath>
#include <type_traits>
#include <concepts>

template<typename Vec>
using Scalar = typename std::decay<decltype(Vec()[0])>::type;

template<typename Vec>
concept FloatVec =
  std::floating_point<Scalar<Vec>> &&
  requires(Vec vec){
    {vec.size()} -> std::integral;
  };

template<FloatVec Vec>
auto norm(const Vec& vec) -> Scalar<Vec> {
  using Size = decltype(vec.size());

  Scalar<Vec> result = 0;
  for(Size i = 0; i < vec.size(); i++){
    result += vec[i] * vec[i];
  }
  return std::sqrt(result);
};

struct Point2 {
  float x;
  float y;

  auto size() const -> int{
    return 2;
  }
  auto operator[](int i) const -> float {
    return i == 0 ? x : y;
  }
};

int main() {
  std::vector<double> a = {3,4};
  std::cout << norm(a) << std::endl;

  Point2 b = {3, 4};
  std::cout << norm(b) << std::endl;

  //std::vector<std::string> c = {"Hello","World"};
  //std::cout << norm(c) << std::endl;

}