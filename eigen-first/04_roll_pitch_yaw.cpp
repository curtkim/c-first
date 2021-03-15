#include <iostream>

#include <Eigen/Core>
#include <Eigen/Geometry>

Eigen::Quaterniond euler2Quaternion(const double roll, const double pitch, const double yaw ){
  using namespace Eigen;

  AngleAxisd rollAngle(roll, Vector3d::UnitX());
  AngleAxisd pitchAngle(pitch, Vector3d::UnitY());
  AngleAxisd yawAngle(yaw, Vector3d::UnitZ());

  Quaterniond q = yawAngle * pitchAngle * rollAngle;
  return q;
}

void assertEquals(Eigen::Vector3d a, Eigen::Vector3d b){
  assert( (a - b).norm() < 0.00001 );
}

int main() {

  using namespace Eigen;

  // 1. euler -> quanterion -> rotate
  Quaterniond quaterniond = euler2Quaternion(0, 0, M_PI/2);
  std::cout << quaterniond.matrix() << std::endl;
  std::cout << "(" << quaterniond.matrix().rows() << "," << quaterniond.matrix().cols() << ")" << std::endl;

  std::cout << "==================" << std::endl;
  Vector3d v(1,0,0);
  assertEquals(Vector3d(0,1,0), quaterniond.matrix() * v);

  // 2. quanterion -> euler
  auto euler = quaterniond.toRotationMatrix().eulerAngles(0, 1, 2);
  std::cout << "Euler from quaternion in roll, pitch, yaw"<< std::endl << euler << std::endl;
  assertEquals(Vector3d(0,0,M_PI/2), euler);

  return 0;
}