#include <iostream>
#include <random>
#include <iterator>
#include <algorithm>

int main() {
  std::vector<int> v { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
  std::vector<int> out;
  std::sample(v.begin(),               // range start
              v.end(),                 // range end
              std::back_inserter(out), // where to put it
              3,                       // number of elements to sample
              std::mt19937{std::random_device{}()});

  std::cout << "Sampled values: ";
  for (const auto &i : out)
    std::cout << i << ", ";
}