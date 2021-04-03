#include <assert.h>

#include <zug/transduce.hpp>
#include <zug/transducer/map.hpp>
#include <zug/transducer/take.hpp>
#include <zug/util.hpp>

#include <vector>


int main()
{
    auto v = std::vector<int>{1, 2, 3, 6};
    assert(transduce(identity, std::plus<int>{}, 1, v) == 13);
}