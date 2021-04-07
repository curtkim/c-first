#include <assert.h>
#include <vector>
#include "ring-span.hpp"


int main() {
  std::vector<int> myVec{1, 2, 3, 4, 5};

  ring_span<int> span1(myVec.data(), myVec.data()+myVec.size(), myVec.data(), 2);
  assert(span1.size() == 2);
  assert(span1[0] == 1);
  assert(span1[1] == 2);

}