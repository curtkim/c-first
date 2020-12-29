#include <iostream>
#include <magic_enum.hpp>

enum DayOfWeek {
  Sunday = 0,
  Monday,
  Tuesday,
  Wednesday,
  Thursday,
  Friday,
  Saturday
};

int main() {
  enum DayOfWeek dow = Friday;
  std::cout << magic_enum::enum_name(dow) << std::endl;

  auto dow2 = magic_enum::enum_cast<DayOfWeek>("Monday");
  std::cout << *dow2 << std::endl;

}