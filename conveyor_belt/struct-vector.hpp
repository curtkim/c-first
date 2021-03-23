#pragma once

//#include <iostream>

namespace temp {
struct LidarData {
  int value;
  int etc;

  LidarData() {
    //std::cout << "LidarData constructor\n";
  }

  LidarData(int value, int etc) : value(value), etc(etc) {
  }
};

struct LidarData2 {
  int value;
  int etc;

  LidarData2(int value, int etc) : value(value), etc(etc) {
  }
};

struct Header {
  int seq;
};
}