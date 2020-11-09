#include <optional>
#include <vector>
#include <assert.h>

#include "pipes/filter.hpp"
#include "pipes/override.hpp"
#include "pipes/push_back.hpp"

void filter() {
  std::vector<int> input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
  auto const ifIsEven = pipes::filter([](int i){ return i % 2 == 0; });

  std::vector<int> expected = {2, 4, 6, 8, 10};

  std::vector<int> results;
  std::copy(begin(input), end(input), ifIsEven >>= pipes::push_back(results));
  assert(results == expected);
}

void iterator_categrory() {
  auto const isEven = pipes::filter([](int i) { return i % 2 == 0; });
  std::vector<int> output;
  static_assert(
    std::is_same<decltype(isEven >>= pipes::push_back(output))::iterator_category, std::output_iterator_tag>::value
    ,"iterator category should be std::output_iterator_tag");
}

void send() {
  std::vector<int> results1;
  auto predicate = [](int i){ return i > 0; };
  auto pipeline = pipes::filter(predicate) >>= pipes::push_back(results1);

  send(1, pipeline);
  assert(results1.size() == 1);
}

int main() {
  filter();
  iterator_categrory();
  send();
  return 0;
}