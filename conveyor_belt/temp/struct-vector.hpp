#pragma once

#include <iostream>

namespace temp {
struct LidarData {
  int value;
  int etc;

  LidarData() {
    //std::cout << "LidarData default constructor\n";
  }

  LidarData(int value, int etc) : value(value), etc(etc) {
    //std::cout << "LidarData constructor\n";
  }
};

struct LidarData2 {
  int value;
  int etc;

  LidarData2(int value, int etc) : value(value), etc(etc) {
    //std::cout << "LidarData2 constructor\n";
  }
};

struct Header {
  int seq;
};
}