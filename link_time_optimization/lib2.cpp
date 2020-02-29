#include "lib2.h"
#include <range/v3/algorithm/count.hpp> // specific includes

int count(std::vector<int> v, int target){
    return ranges::count(v, target);
}