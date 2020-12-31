#include <type_traits>
#include <cstdint>

template <typename T>
auto interpolate(T left, T right, T power) -> std::enable_if_t<std::is_floating_point_v<T>, T> {
  return left * (1 - power) + right * power;
}

int main() {
  interpolate(1.0, 2.0, 1.5);
  //interpolate(0, 100, 0); not enabled
}