#include <numeric>

constexpr int naiveSum(unsigned int n) {
  auto p = new int[n];
  std::iota(p, p+n, 1);
  auto tmp = std::accumulate(p, p+n, 0);
  delete[] p;
  return tmp;
}

constexpr int smartSum(unsigned int n) {
  return (n*(1+n))/2;
}

int main() {
  static_assert(naiveSum(10) == smartSum(10));
  static_assert(naiveSum(11) == smartSum(11));
  return 0;
}