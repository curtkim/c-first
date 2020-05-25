#include <iostream>

#include <Eigen/Core>
#include <Eigen/Geometry>

Eigen::Quaterniond euler2Quaternion(const double roll, const double pitch, const double yaw ){
  Eigen::AngleAxisd rollAngle(roll, Eigen::Vector3d::UnitX());
  Eigen::AngleAxisd pitchAngle(pitch, Eigen::Vector3d::UnitY());
  Eigen::AngleAxisd yawAngle(yaw, Eigen::Vector3d::UnitZ());

  Eigen::Quaterniond q = yawAngle * pitchAngle * rollAngle;
  return q;
}

void assertEquals(Eigen::Vector3d a, Eigen::Vector3d b){
  assert( (a - b).norm() < 0.00001 );
}

int main() {

  // 1. euler -> quanterion -> rotate
  Eigen::Quaterniond quaterniond = euler2Quaternion(0, 0, M_PI/2);
  std::cout << quaterniond.matrix() << std::endl;
  std::cout << "(" << quaterniond.matrix().rows() << "," << quaterniond.matrix().cols() << ")" << std::endl;

  std::cout << "==================" << std::endl;
  Eigen::Vector3d v(1,0,0);
  assertEquals(Eigen::Vector3d(0,1,0), quaterniond.matrix() * v);

  // 2. quanterion -> euler
  auto euler = quaterniond.toRotationMatrix().eulerAngles(0, 1, 2);
  std::cout << "Euler from quaternion in roll, pitch, yaw"<< std::endl << euler << std::endl;
  assertEquals(Eigen::Vector3d(0,0,M_PI/2), euler);

  return 0;
}