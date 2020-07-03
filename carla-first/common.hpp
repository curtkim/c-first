
#ifndef CARLA_FRIST_COMMON_HPP
#define CARLA_FRIST_COMMON_HPP

inline bool ends_with(std::string const & value, std::string const & ending)
{
  if (ending.size() > value.size()) return false;
  return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

long getEpochMillisecond() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

long getEpochMicrosecond() {
  using namespace std::chrono;
  return duration_cast<microseconds>(system_clock::now().time_since_epoch()).count();
}

long getEpochNanosecond() {
  using namespace std::chrono;
  return duration_cast<nanoseconds>(system_clock::now().time_since_epoch()).count();
}

#endif // CARLA_FRIST_COMMON_HPP
