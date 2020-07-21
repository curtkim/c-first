#include <iostream>
#include <vector>
#include <gsl/span>
#include <fmt/format.h>


int64_t add(gsl::span<int> s) {
  int64_t sum = 0;
  for(auto i : s) //iterate over s
    sum += i;
  return sum;
}

int main(int argc, char** argv)
{
  /* An array. The size is implicit. */
  int iArr[] = {1,1,1,1};
  std::cout << add(iArr) << "\n"; //4

  /* A dynamic array. Requires explicit pointer and size to construct span. */
  int* iPtr = new int[4];
  iPtr[0] = iPtr[1] = iPtr[2] = iPtr[3] = 1;
  std::cout << add({iPtr, 2}) << "\n"; //4

  /* An std::array */
  std::array<int, 4> isArr = {1,1,1,1};
  std::cout << add(isArr) << "\n"; //4

  /* A std::vector. Requires explicit pointer and size to construct span.*/
  std::vector<int> iVec = {1,1,1,1};
  std::cout << add({&iVec[0], 2}) << "\n"; //4
  return 0;
}

