#include <iostream>
#include <iomanip>
#include <ctime>
#include <chrono>

int main()
{
  using namespace std::chrono;

  system_clock::time_point now = system_clock::now();
  duration<double> d = duration<double>(now.time_since_epoch());

  std::cout << std::setprecision (std::numeric_limits<double>::digits10 + 1)
    << d.count() << std::endl;


  std::time_t t_c = system_clock::to_time_t(now - hours(24));
  std::cout << "24 hours ago, the time was "
            << std::put_time(std::localtime(&t_c), "%F %T") << '\n';

  steady_clock::time_point start = steady_clock::now();
  std::cout << "Hello World\n";
  steady_clock::time_point end = steady_clock::now();
  std::cout << "Printing took "
            << duration_cast<std::chrono::microseconds>(end - start).count()
            << "us.\n";

  std::cout << "sizeof(start)=" << sizeof(start) << std::endl;
  std::cout << "sizeof(std::chrono::system_clock::time_point)=" << sizeof(std::chrono::system_clock::time_point) << std::endl;
}