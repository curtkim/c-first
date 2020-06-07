#include <iostream>

#include <chrono>  // chrono::system_clock
#include <ctime>   // localtime
#include <sstream> // stringstream
#include <iomanip> // put_time
#include <string>  // string

#include <fmt/chrono.h>


long getEpochMillisecond() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

int main()
{
  // 1. std::time_t : epoch time second
  std::time_t t = std::time(nullptr);
  std::cout << t << " seconds since the Epoch\n";

  // 2. time_t -> tm(localtime)
  std::tm localtime = *std::localtime(&t);
  fmt::print("The date is {:%Y-%m-%d %H:%M:%S}", localtime);
  std::cout << std::endl;

  // 3. epoch milli second
  long epochMillisecond = getEpochMillisecond();
  std::cout << epochMillisecond << " seconds since the Epoch\n";
  std::cout << "millisecond only = " << epochMillisecond % 1000 << std::endl;
  // padded with zero
  fmt::print("{:#03}\n", epochMillisecond % 1000);

  // 4. format put_time
  std::cout << std::put_time(&localtime, "%Y-%m-%d %X") << std::endl;

  return 0;
}