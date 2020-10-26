#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>
#include <assert.h>

using u64_millis = std::chrono::duration<uint64_t, std::chrono::milliseconds>;

/*
static std::chrono::time_point<std::chrono::system_clock, u64_millis> u64_to_time(uint64_t timestamp) {
  return std::chrono::time_point<std::chrono::system_clock, u64_millis>{u64_millis{timestamp}};
}
*/

void duration_int_milli() {
  // integral representation of 10 milliseconds
  std::chrono::duration<int, std::milli> d(10);
  assert(d.count() == 10);

  d = std::chrono::milliseconds(5);
  assert(d.count() == 5);

  // Casting from seconds to milliseconds can happen implicitly
  d = std::chrono::seconds(10);
  assert(d.count() == 10000);
}

void duration_long_milli() {
  // integral representation of 10 milliseconds
  std::chrono::duration<long, std::milli> d(10);
  assert(d.count() == 10);

  d = std::chrono::milliseconds(5);
  assert(d.count() == 5);

  // Casting from seconds to milliseconds can happen implicitly
  d = std::chrono::seconds(10);
  assert(d.count() == 10000);
}

void duration_long_nano() {
  // integral representation of 10 nanoseconds
  std::chrono::duration<unsigned long, std::nano> d(10);
  assert(d.count() == 10);

  d = std::chrono::milliseconds(5);
  assert(d.count() == 5*1000*1000);

  // Casting from seconds to milliseconds can happen implicitly
  d = std::chrono::seconds(1);
  assert(d.count() == 1000*1000*1000);
}

void duration_literal() {
  using namespace std::chrono_literals;

  // integral rep of 1 second
  std::chrono::duration<int> d1 = 1s;
  assert(d1.count() == 1);

  // floating-point rep of 1 second
  std::chrono::duration<float> d2 = 1s;
  assert(d2.count() == 1.f);

  // floating-point rep of a fraction of a second
  std::chrono::duration<float> d3 = 1ms;
  assert(d3.count() == 0.001f);
}

int main() {
  duration_int_milli();
  duration_long_milli();
  duration_long_nano();
  duration_literal();
}
