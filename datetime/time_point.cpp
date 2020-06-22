#include <iostream>
#include <iomanip>
#include <ctime>
#include <chrono>

int main()
{
  std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
  std::chrono::duration<double> d = std::chrono::duration<double>(now.time_since_epoch());

  std::cout << std::setprecision (std::numeric_limits<double>::digits10 + 1)
    << d.count() << std::endl;


  std::time_t t_c = std::chrono::system_clock::to_time_t(now - std::chrono::hours(24));
  std::cout << "24 hours ago, the time was "
            << std::put_time(std::localtime(&t_c), "%F %T") << '\n';

  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  std::cout << "Hello World\n";
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  std::cout << "Printing took "
            << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
            << "us.\n";

  std::cout << "sizeof(start)=" << sizeof(start) << std::endl;
  std::cout << "sizeof(std::chrono::system_clock::time_point)=" << sizeof(std::chrono::system_clock::time_point) << std::endl;
}