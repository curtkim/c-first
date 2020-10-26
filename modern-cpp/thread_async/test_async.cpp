#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <future>
#include <thread>


template <typename RandomIt>
int parallel_sum(RandomIt beg, RandomIt end)
{
  auto len = end - beg;
  if (len < 1000)
    return std::accumulate(beg, end, 0);

  RandomIt mid = beg + len/2;
  std::cerr << std::this_thread::get_id() <<  " async " << std::endl;
  auto handle = std::async(std::launch::async,
                           parallel_sum<RandomIt>, mid, end);
  int sum = parallel_sum(beg, mid);
  return sum + handle.get();
}

int main() {
  std::vector<int> v(10000, 1);
  std::cout << "The sum is " << parallel_sum(v.begin(), v.end()) << '\n';
  return 0;
}