#include <iostream>
#include <magic_enum.hpp>

enum class LANES {
  LIDAR1,
  CAMERA1,
  CAMERA2,
  CAMERA3,
  GPS,
  IMU,
};

int main() {
  assert(0 == magic_enum::enum_integer(LANES::LIDAR1));
  assert(1 == magic_enum::enum_integer(LANES::CAMERA1));
  assert(5 == magic_enum::enum_integer(LANES::IMU));
}