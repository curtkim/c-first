#include <thrust/device_vector.h>
#include <iostream>

int main(void)
{
  // allocate three device_vectors with 10 elements
  thrust::device_vector<int> X(10);

  // initialize X to 0,1,2,3, ....
  thrust::sequence(X.begin(), X.end());

  int sum = thrust::reduce(X.begin(), X.end(), (int) 0, thrust::plus<int>());
  std::cout << sum << std::endl;

  // 1이 몇번 나오는가?
  int count = thrust::count(X.begin(), X.end(), 1);
  std::cout << count << std::endl;

  return 0;
}