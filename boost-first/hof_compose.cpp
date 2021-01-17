#include <vector>
#include <iostream>
#include <boost/hof/compose.hpp>

int plusOne(int i)
{
  return i + 1;
}

int timesTwo(int i)
{
  return i * 2;
}


int main() {
  auto const input = std::vector<int>{1, 2, 3, 4, 5};
  auto results = std::vector<int>{};

  std::transform(begin(input), end(input), back_inserter(results), boost::hof::compose(timesTwo, plusOne));

  for(const auto & v : results){
    std::cout << v << std::endl;
  }
}