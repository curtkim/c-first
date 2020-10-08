#include <iostream>
#include <array>

constexpr int factorial(int n) {
  return n == 0 ? 1 : n * factorial(n - 1);
}

/*
template <typename T, std::size_t N, typename F>
constexpr std::array<std::result_of_t<F(T)>, N> transform(std::array<T, N> array, F f) {
  std::array<T, N> result;
  for(int i = 0; i < N; i++){
    result[i] = F(array[i]);
  }
  return result;
}
*/

constexpr std::array<int, 4> ints{{1, 2, 3, 4}};
constexpr std::array<int, 4> facts2{{factorial(1), factorial(2), factorial(3), factorial(4)}};
//constexpr std::array<int, 4> facts = transform(ints, factorial);


int main() {
  std::cout << facts2[3] << std::endl;
}