//
// Created by curt on 20. 6. 22..
//

#ifndef CARLA_FRIST_70_HEADER_HPP
#define CARLA_FRIST_70_HEADER_HPP

#include <cstdint>

struct Header{
  uint64_t frame;
  uint64_t timepoint;
  uint16_t record_type;
  uint16_t topic_name_length;
  uint32_t body_length;
  uint32_t param1;
  uint32_t param2;
};

std::ostream& operator<<(std::ostream& os, const Header& header)
{
  return os << "frame=" << header.frame
    << " time=" << header.timepoint
    << " type=" << header.record_type;
}

#endif //CARLA_FRIST_70_HEADER_HPP
