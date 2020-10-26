#include <iostream>
#include <vector>
#include <numeric>
#include <chrono>
#include <assert.h>
#include <iomanip>

using namespace std::chrono;

using u64_millis = std::chrono::duration<uint64_t, std::chrono::milliseconds>;

/*
static std::chrono::time_point<std::chrono::system_clock, u64_millis> u64_to_time(uint64_t timestamp) {
  return std::chrono::time_point<std::chrono::system_clock, u64_millis>{u64_millis{timestamp}};
}
*/

void print_tp(time_point<system_clock>& tp) {
  std::time_t t_c = system_clock::to_time_t(tp);
  std::cout << "the time was " << std::put_time(std::localtime(&t_c), "%F %T") << '\n';
}

void timepoint(){
  time_point<system_clock> t = system_clock::now();
  // 1_597_034_593_815_536_937
  std::cout << t.time_since_epoch().count() << std::endl;
  assert(sizeof(t.time_since_epoch().count()) == 8);
  std::cout << typeid(t).name() << std::endl;
}

void by_uint64_millisecond() {
  milliseconds d(1597034593815);
  time_point<system_clock> tp{d};
  print_tp(tp);
}

void by_uint64_nanosecond() {
  nanoseconds d(1597034593815536937);
  time_point<system_clock> tp{d};
  print_tp(tp);
}

void by_uint32_second() {
  using u32_sec = std::chrono::duration<uint32_t>;

  u32_sec d(1597034593);
  time_point<system_clock> tp{d};
  print_tp(tp);
}

void by_uint32_hour() {
  using u32_hours = std::chrono::duration<uint32_t, std::ratio<3600>>;

  u32_hours d(443620);
  time_point<system_clock> tp{d};
  print_tp(tp);
}

void by_uint16_day() {
  using u16_day = std::chrono::duration<uint32_t, std::ratio<24*3600>>;

  u16_day d(18484);
  time_point<system_clock> tp{d};
  print_tp(tp);
}

void by_cast() {
  std::chrono::duration<double> d = 1597034593815.536937ms;
  milliseconds d2 = std::chrono::duration_cast<std::chrono::milliseconds>(d);

  // time_point를 double duration으로 생성할 수 없다.
  time_point<system_clock> tp{d2};
  print_tp(tp);
}


int main() {
  by_uint64_millisecond();
  by_uint64_nanosecond();
  by_uint32_second();
  by_uint32_hour();
  by_uint16_day();
  by_cast();

  timepoint();
}
