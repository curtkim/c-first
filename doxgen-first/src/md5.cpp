#include "Poco/DigestStream.h"
#include "Poco/MD5Engine.h"

#include "transport/car.hpp"
#include <iostream>

int main(int argc, char **argv) {
  Poco::MD5Engine md5;
  Poco::DigestOutputStream ds(md5);
  ds << "abcdefghijklmnopqrstuvwxyz";
  ds.close();
  std::cout << Poco::DigestEngine::digestToHex(md5.digest()) << std::endl;

  transport::Car car{"abc", 2010};
  car.Print();

  return 0;
}
