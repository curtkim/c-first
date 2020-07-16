#include <iostream>
#include <vector>

#include <tbb/tbb.h>
#include <pstl/execution>
#include <pstl/algorithm>
#include <pstl/numeric>

//
// For best performance when using the Intel compiler use
// EXTRA_CXXFLAGS="-qopenmp-simd -xHost" when building

const int num_versions = 5;

static void warmupTBB();
void accumulateTime(tbb::tick_count& t0, int version);
void dumpTimes();

void fig_4_11() {
  const int num_trials = 1000;
  const int n = 65536;
  tbb::tick_count t0;

  std::vector<float> a(n, 1.0);

  for (int t = 0; t < num_trials; ++t) {
    warmupTBB();
    t0 = tbb::tick_count::now();
    float sum = std::reduce(pstl::execution::par, a.begin(), a.end());
    accumulateTime(t0, 3);
    sum += std::reduce(pstl::execution::par_unseq, a.begin(), a.end());
    accumulateTime(t0, 4);
#pragma novector
    for (int i = 0; i < n; ++i) {
      sum += a[i];
    }
    accumulateTime(t0, 0);
    sum += std::reduce(pstl::execution::seq, a.begin(), a.end());
    accumulateTime(t0, 1);
    sum += std::reduce(pstl::execution::unseq, a.begin(), a.end());
    accumulateTime(t0, 2);
    if (sum != num_versions * n)
      std::cout << "ERROR: sum is not correct"
                << sum << " != " << num_versions * n << std::endl;
  }
  dumpTimes();
}

void fig_4_11_with_lambda() {
  const int num_trials = 1000;
  const int n = 65536;
  tbb::tick_count t0;

  std::vector<float> a(n, 1.0);

  for (int t = 0; t < num_trials; ++t) {
    warmupTBB();
    t0 = tbb::tick_count::now();
    auto sum = std::reduce(pstl::execution::par,
      /* in1 range */ a.begin(), a.end(),
      /* init */ 0.0,
                           [](float ae, float be) -> float {
                             return ae + be;
                           }
    );
    accumulateTime(t0, 3);
    sum += std::reduce(pstl::execution::par_unseq,
      /* in1 range */ a.begin(), a.end(),
      /* init */ 0.0,
                       [](float ae, float be) -> float {
                         return ae + be;
                       }
    );
    accumulateTime(t0, 4);
#pragma novector
    for (int i = 0; i < n; ++i) {
      sum += a[i];
    }
    accumulateTime(t0, 0);
    sum += std::reduce(pstl::execution::seq,
      /* in1 range */ a.begin(), a.end(),
      /* init */ 0.0,
                       [](float ae, float be) -> float {
                         return ae + be;
                       }
    );
    accumulateTime(t0, 1);
    sum += std::reduce(pstl::execution::unseq,
      /* in1 range */ a.begin(), a.end(),
      /* init */ 0.0,
                       [](float ae, float be) -> float {
                         return ae + be;
                       }
    );
    accumulateTime(t0, 2);
    if (sum != num_versions * n)
      std::cout << "ERROR: sum is not correct"
                << sum << " != " << num_versions * n << std::endl;
  }
  dumpTimes();
}

double total_times[num_versions] = {0,0,0,0,0};

void accumulateTime(tbb::tick_count& t0, int version) {
  if (version >= 0) {
    double elapsed_time = (tbb::tick_count::now()-t0).seconds();
    total_times[version] += elapsed_time;
    t0 = tbb::tick_count::now();
  }
}

void dumpTimes() {
  const char *versions[num_versions] =
    { "for", "seq", "unseq", "par", "par_unseq" };

  for (int i = 0; i < num_versions; ++i) {
    std::cout << versions[i] << ", " << total_times[i] << std::endl;
    total_times[i] = 0.0;
  }
}

static void warmupTBB() {
  tbb::parallel_for(0, tbb::task_scheduler_init::default_num_threads(), [](int) {
    tbb::tick_count t0 = tbb::tick_count::now();
    while ((tbb::tick_count::now() - t0).seconds() < 0.01);
  });
}

int main() {
  double  total_time = 0.0;
  {
    tbb::tick_count t0 = tbb::tick_count::now();
    fig_4_11();
    fig_4_11_with_lambda();
    total_time = (tbb::tick_count::now() - t0).seconds();
  }
  std::cout << "total_time == " << total_time << " seconds" << std::endl;
  return 0;
}