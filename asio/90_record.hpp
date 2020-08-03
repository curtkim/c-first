#pragma once

#include <iostream>
#include <vector>

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

struct Record {
  Header header;
  std::string topic_name;
  std::vector<char> body;

  Record(Header header, std::string topic_name, std::vector<char> body) : header(header), topic_name(topic_name), body(body) {}
};