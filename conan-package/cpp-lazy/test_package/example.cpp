#include <assert.h>
#include <Lz/Filter.hpp>
#include <vector>

int main()
{
	std::vector<int> toFilter = {1, 2, 3, 4, 5, 6};
  const auto filter = lz::filter(toFilter, [](const int i) { return i % 2 == 0; });

	assert(filter[0] == 2);
	assert(filter[1] == 4);
	assert(filter[2] == 6);
}
