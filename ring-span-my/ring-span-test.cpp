#include <assert.h>
#include <vector>
#include "ring-span.hpp"
#include <stdio.h>

int main() {
  std::vector<int> myVec{1, 2, 3, 4, 5};

  {
    ring_span<int> span1(myVec.data(), myVec.data() + myVec.size(), myVec.data(), 2);
    assert(span1.size() == 2);
    assert(span1[0] == 1);
    assert(span1[1] == 2);

    for (auto a : span1)
      printf("%d ", a);
    printf("\n");
  }

  {
    ring_span<int> span2(myVec.data(), myVec.data() + myVec.size(), myVec.data() + 3, 3);
    assert(span2.size() == 3);
    assert(span2[0] == 4);
    assert(span2[1] == 5);
    assert(span2[2] == 1);

    for (auto a : span2)
      printf("%d ", a);
  }
}