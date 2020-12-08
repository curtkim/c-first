#include "common.h"

int main()
{
  using nonstd::ring_span;

  double arr[] = { 0, 1, 2, };
  ring_span<double> span( arr, arr + dim(arr) );

  // empty?
  std::cout << span << "\n";
}