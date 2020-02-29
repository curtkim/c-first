#include "lib1.h"
#include <range/v3/all.hpp> // get everything

int sum(std::vector<int> v){
    return ranges::accumulate(v, 0); // 12.2
}