#include <vector>
#include <iostream>
#include <algorithm>
#include <random>
#include <tbb/tick_count.h>
#include <tbb/parallel_for.h>
#include <tbb/task_scheduler_init.h>
#include <tbb/atomic.h>
#include <tbb/cache_aligned_allocator.h>

struct atom_bin {
  alignas(128) tbb::atomic<int> count;
};

int main(int argc, char **argv) {

  long int n = 1000 * 1000 * 100;
  int nth = 4;
  constexpr int num_bins = 256;

  // Initialize random number generator
  std::random_device seed;    // Random device seed
  std::mt19937 mte{seed()};   // mersenne_twister_engine
  std::uniform_int_distribution<> uniform{0, num_bins};
  // Initialize image
  std::vector<uint8_t> image; // empty vector
  image.reserve(n);           // image vector prealocated
  std::generate_n(std::back_inserter(image), n,
                  [&] { return uniform(mte); }
  );
  // Initialize histogram
  std::vector<int> hist(num_bins);

  tbb::task_scheduler_init init{nth};

  // Serial execution
  tbb::tick_count t0 = tbb::tick_count::now();
  std::for_each(image.begin(), image.end(),
                [&](uint8_t i) { hist[i]++; });
  tbb::tick_count t1 = tbb::tick_count::now();
  double t_serial = (t1 - t0).seconds();
  std::cout << "Serial: " << t_serial << ", ";


  // Parallel execution
  std::vector<atom_bin, tbb::cache_aligned_allocator<atom_bin>> hist_p(num_bins);
  t0 = tbb::tick_count::now();
  parallel_for(
    tbb::blocked_range<size_t>{0, image.size()},
    [&](const tbb::blocked_range<size_t> &r) {
      std::for_each(image.data() + r.begin(), image.data() + r.end(),
                    [&](const int i) { hist_p[i].count++; });
    });
  t1 = tbb::tick_count::now();
  double t_parallel = (t1 - t0).seconds();


  std::cout << "Parallel: " << t_parallel << ", ";
  std::cout << "Speed-up: " << t_serial / t_parallel << std::endl;

  for (size_t idx = 0; idx < hist_p.size(); ++idx) {
    if (hist[idx] != hist_p[idx].count) {
      std::cerr << "Index "
                << idx
                << " failed: "
                << hist[idx] << " != " << hist_p[idx].count
                << std::endl;
      return -1;
    }
  }
  return 0;
}