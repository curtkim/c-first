#include <optional>
#include <iostream>
#include <vector>
#include "pipes/filter.hpp"
#include "pipes/transform.hpp"
#include "pipes/push_back.hpp"


int main() {
  auto const source = std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  auto destination = std::vector<int>{};

  source  >>= pipes::filter([](int i){ return i % 2 == 0; })
          >>= pipes::transform([](int i){ return i * 2; })
          >>= pipes::push_back(destination);

  std::cout << destination[0] << std::endl;
}