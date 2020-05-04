
#ifndef CARLA_FRIST_COMMON_HPP
#define CARLA_FRIST_COMMON_HPP

inline bool ends_with(std::string const & value, std::string const & ending)
{
  if (ending.size() > value.size()) return false;
  return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

#endif // CARLA_FRIST_COMMON_HPP
