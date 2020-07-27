#include <vector>
#include <iostream>
#include <algorithm>
#include <random>
#include <tbb/tick_count.h>

int main() {
  constexpr long int n = 1000 * 1000;
  constexpr int num_bins = 256;

  // Initialize random number generator
  std::random_device seed;    // Random device seed
  std::mt19937 mte{seed()};   // mersenne_twister_engine
  std::uniform_int_distribution<> uniform{0, num_bins};

  // Initialize image
  std::vector<uint8_t> image; // empty vector
  image.reserve(n);           // image vector prealocated
  std::generate_n(
    std::back_inserter(image),
    n,
    [&] { return uniform(mte); }
  );

  // Initialize histogram
  std::vector<int> hist(num_bins);

  // Serial execution
  tbb::tick_count t0 = tbb::tick_count::now();
  std::for_each(
    image.begin(), image.end(),
    [&hist](uint8_t i) { hist[i]++; }
  );
  tbb::tick_count t1 = tbb::tick_count::now();
  double t_serial = (t1 - t0).seconds();

  std::cout << "Serial time: " << t_serial << std::endl;
  return 0;
}