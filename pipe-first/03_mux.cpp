#include <optional>
#include <iostream>
#include <vector>
#include "pipes/filter.hpp"
#include "pipes/transform.hpp"
#include "pipes/push_back.hpp"
#include "pipes/mux.hpp"

#include <assert.h>

int main() {
  auto const input1 = std::vector<int>{1, 2, 3, 4, 5};
  auto const input2 = std::vector<int>{10, 20, 30, 40, 50};

  auto results = std::vector<int>{};

  pipes::mux(input1, input2) >>= pipes::filter   ([](int a, int b){ return a + b < 40; })
    >>= pipes::transform([](int a, int b) { return a * b; })
    >>= pipes::push_back(results);

  auto const expected = std::vector<int>{10,40,90};
  assert(results == expected);
}