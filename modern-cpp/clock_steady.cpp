#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>

volatile int sink;
int main()
{
  for (auto size = 1ull; size < 1000000000ull; size *= 100) {
    // record start time
    auto start = std::chrono::steady_clock::now();

    // do some work
    std::vector<int> v(size, 42);
    sink = std::accumulate(v.begin(), v.end(), 0u); // make sure it's a side effect

    // record end time
    auto end = std::chrono::steady_clock::now();

    // duration
    std::chrono::duration<double> diff = end-start;

    std::cout << "Time to fill and iterate a vector of "
              << size << " ints : " << diff.count() << " s\n";
  }
}