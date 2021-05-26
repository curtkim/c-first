#pragma once

#include <chrono>

class StopWatch {
  using clock_t = std::chrono::high_resolution_clock;

public:
  StopWatch(): start_time(clock_t::now()) { }

  long get_elapsed_time() const {
      return (std::chrono::high_resolution_clock::now() - start_time).count();
  }

  void reset() {
      start_time = clock_t::now();
  }

private:
  clock_t::time_point start_time;
};