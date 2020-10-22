#include <iostream>
#include <algorithm>  // clamp
#include <numeric>    // for gcm, lcm

int main() {

  std::cout << std::clamp(300, 0, 255) << std::endl;
  std::cout << std::clamp(-10, 0, 255) << std::endl;

  std::cout << std::gcd(24, 60) << std::endl;
  std::cout << std::lcm(15, 50) << std::endl;
}