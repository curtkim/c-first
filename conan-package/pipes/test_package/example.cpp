#include <cassert>
#include <iostream>
#include <algorithm>
#include <vector>

#include "pipes/filter.hpp"
#include "pipes/override.hpp"
#include "pipes/push_back.hpp"

int main()
{
    std::vector<int> input = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    auto const ifIsEven = pipes::filter([](int i){ return i % 2 == 0; });
    
    std::vector<int> expected = {2, 4, 6, 8, 10};
    
    std::vector<int> results;
    std::copy(begin(input), end(input), ifIsEven >>= pipes::push_back(results));

    assert(results == expected);
    return 0;
}
