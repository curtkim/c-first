#include <iostream>
#include <fstream>
#include "info.pb.h"

int main() {

  Info info;
  std::ifstream ifs("info.data");
  if (!info.ParseFromIstream(&ifs)) {
    std::cout << "failed" << std::endl;
  }

  info.PrintDebugString();
  std::cout << info.ByteSize() << std::endl;

  std::cout << info.bytes().length() << std::endl;
  std::cout << info.bytes() << std::endl;

  return 0;
}