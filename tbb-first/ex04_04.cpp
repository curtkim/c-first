#include <iostream>
#include <vector>

#include <tbb/tbb.h>
#include <pstl/execution>
#include <pstl/algorithm>

//
// For best performance when using the Intel compiler use
// EXTRA_CXXFLAGS="-qopenmp-simd -xHost" when building

enum Policy {
  forloop,
  none,
  seq,
  unseq,
  par,
  par_unseq,
  pfor,
};

inline void f(float& i) { i += 1; }

void warmupTBB();
void accumulateTime(tbb::tick_count& t0, int version);
void validateResults(int num_trials, const std::vector<float>& v);

void fig_4_4() {
  const int num_trials = 10000;
  const int n = 65536;
  tbb::tick_count t0;

  std::vector<float> v(n, 0);

  for (int t = 0; t < num_trials; ++t) {
    warmupTBB();
    t0 = tbb::tick_count::now();
    std::for_each(pstl::execution::par, v.begin(), v.end(),
                  [](float& i) {
                    f(i);
                  });
    accumulateTime(t0, par);

    std::for_each(pstl::execution::par_unseq, v.begin(), v.end(),
                  [](float& i) {
                    f(i);
                  });
    accumulateTime(t0, par_unseq);

    tbb::parallel_for(0, (int)v.size(),
                      [&v](int i) {
                        f(v[i]);
                      });
    accumulateTime(t0, pfor);
#pragma novector
    for (auto& i : v ) {
      f(i);
    }
    accumulateTime(t0, forloop);

    std::for_each(v.begin(), v.end(),
                  [](float& i) {
                    f(i);
                  });
    accumulateTime(t0, none);

    std::for_each(pstl::execution::seq, v.begin(), v.end(),
                  [](float& i) {
                    f(i);
                  });
    accumulateTime(t0, seq);

    std::for_each(pstl::execution::unseq, v.begin(), v.end(),
                  [](float& i) {
                    f(i);
                  });
    accumulateTime(t0, unseq);
  }
  validateResults(num_trials, v);
}

const int num_versions = 7;
double total_times[num_versions] = {0,0,0,0,0,0,0};

void accumulateTime(tbb::tick_count& t0, int version) {
  if (version >= 0) {
    double elapsed_time = (tbb::tick_count::now()-t0).seconds();
    total_times[version] += elapsed_time;
    t0 = tbb::tick_count::now();
  }
}

void dumpTimes() {
  const char *versions[num_versions] =
    { "for", "none", "seq", "unseq", "par", "par_unseq", "pfor" };

  for (int i = 0; i < num_versions; ++i)
    std::cout << versions[i] << ", " << total_times[i] << std::endl;
}

void validateResults(int num_trials, const std::vector<float>& v) {
  float r = num_trials * num_versions;
  for (auto& i : v ) {
    if (r != i) {
      std::cout << "ERROR: results did not match" << std::endl;
      return;
    }
  }
}

void warmupTBB() {
  tbb::parallel_for(0, tbb::task_scheduler_init::default_num_threads(), [](int) {
    tbb::tick_count t0 = tbb::tick_count::now();
    while ((tbb::tick_count::now() - t0).seconds() < 0.001);
  });
}

int main() {
  double  total_time = 0.0;
  {
    tbb::tick_count t0 = tbb::tick_count::now();
    fig_4_4();
    total_time = (tbb::tick_count::now() - t0).seconds();
  }
  dumpTimes();
  std::cout << "total_time == " << total_time << " seconds" << std::endl;
  return 0;
}