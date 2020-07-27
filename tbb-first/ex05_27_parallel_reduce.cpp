#include <vector>
#include <iostream>
#include <algorithm>
#include <random>
#include <tbb/tick_count.h>
#include <tbb/task_scheduler_init.h>
#include <tbb/atomic.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>

int main(int argc, char **argv) {

  long int n = 1000*1000*100;
  int nth = 4;
  constexpr int num_bins = 256;

  // Initialize random number generator
  std::random_device seed;    // Random device seed
  std::mt19937 mte{seed()};   // mersenne_twister_engine
  std::uniform_int_distribution<> uniform{0, num_bins};
  // Initialize image
  std::vector<uint8_t> image; // empty vector
  image.reserve(n);           // image vector prealocated
  std::generate_n(
    std::back_inserter(image), n,
    [&] { return uniform(mte); }
  );
  // Initialize histogram
  std::vector<int> hist(num_bins);

  tbb::task_scheduler_init init{nth};

  // Serial execution
  tbb::tick_count t0 = tbb::tick_count::now();
  std::for_each(
    image.begin(), image.end(),
    [&](uint8_t i) { hist[i]++; });
  tbb::tick_count t1 = tbb::tick_count::now();
  double t_serial = (t1 - t0).seconds();

  // Parallel execution
  using vector_t = std::vector<int>;
  using image_iterator = std::vector<uint8_t>::iterator;
  t0 = tbb::tick_count::now();
  vector_t hist_p = parallel_reduce(
    tbb::blocked_range<image_iterator>{image.begin(), image.end()},
    vector_t(num_bins),
    // 1st Lambda: Parallel computation on private histograms
    [](const tbb::blocked_range<image_iterator> &r, vector_t v) {
      std::for_each(r.begin(), r.end(), [&v](uint8_t i) { v[i]++; });
      return v;
    },
    // 2nd Lambda: Parallel reduction of the private histograms
    [num_bins](vector_t a, const vector_t &b) -> vector_t {
      for (int i = 0; i < num_bins; ++i) a[i] += b[i];
      return a;
    });
  t1 = tbb::tick_count::now();
  double t_parallel = (t1 - t0).seconds();

  std::cout << "Serial: " << t_serial << ", ";
  std::cout << "Parallel: " << t_parallel << ", ";
  std::cout << "Speed-up: " << t_serial / t_parallel << std::endl;

  if (hist != hist_p)
    std::cerr << "Parallel computation failed!!" << std::endl;

  return 0;
}