#include <assert.h>
#include <span>
#include <algorithm>
#include <stdio.h>

int main() {
  int a[] { 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 };

  std::span span1{a, 3};
  assert(span1.size() == 3);
  assert(span1.front() == 0);
  assert(span1.back() == 2);

  assert(span1[0] == 0);
  assert(span1[1] == 1);
  assert(span1[2] == 2);
  assert(!span1.empty());

  // 홀수를 찾는다.
  auto it = std::find_if(span1.begin(), span1.end(), [](auto i){
    return i%2 != 0;
  });
  assert(*it == 1);

  // for loop
  for(const auto item : span1){
    printf("%d, ", item);
  }
  printf("\n");

  // begin, end
  printf("%lu %lu", span1.begin(), span1.end());
}