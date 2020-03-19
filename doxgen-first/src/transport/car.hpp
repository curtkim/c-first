//
// Created by curt on 20. 3. 16..
//

#ifndef MD5ENCRYPTER_CAR_HPP
#define MD5ENCRYPTER_CAR_HPP

#include <iostream>

namespace transport {
class Car {

private:
  /*!< an string value */
  std::string name;

  // 출시년도
  int year;

public:
  /**
   * A constructor.
   * A more elaborate description of the constructor.
   */
  Car(const std::string &name, int year) : name(name), year(year) {}

  void Print();
};
} // namespace transport
#endif // MD5ENCRYPTER_CAR_HPP
