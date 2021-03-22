#include <assert.h>
#include <iostream>
#include <iomanip>
#include <ctime>
#include <chrono>

int main()
{
  using namespace std::chrono;

  assert(8 == sizeof(system_clock::time_point));
  assert(8 == sizeof(std::time_t));
  assert(8 == sizeof(long));

  system_clock::time_point now = system_clock::now();
  std::cout << now.time_since_epoch().count() << " " << sizeof(now.time_since_epoch().count()) << "\n";

  system_clock::time_point yesterday = now - hours(24);
  std::time_t t_c = system_clock::to_time_t(yesterday);
  std::cout << "24 hours ago, the time was "
            << std::put_time(std::localtime(&t_c), "%F %T") << '\n';
}