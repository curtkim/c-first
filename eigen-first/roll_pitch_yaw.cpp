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

void quanternion2euler(Eigen::Quaterniond quaterniond) {
  auto euler = quaterniond.toRotationMatrix().eulerAngles(0, 1, 2);
  std::cout << "Euler from quaternion in roll, pitch, yaw"<< std::endl << euler << std::endl;
  assert( (Eigen::Vector3d(0,0,M_PI/2) - euler).norm() < 0.00001 );
}

int main() {
  Eigen::Quaterniond quaterniond = euler2Quaternion(0, 0, M_PI/2);
  std::cout << quaterniond.matrix() << std::endl;

  Eigen::Vector3d v(1,0,0);
  std::cout << quaterniond.matrix() * v << std::endl;
  assert( (Eigen::Vector3d(0,1,0) - quaterniond.matrix() * v).norm() < 0.00001 );

  quanternion2euler(quaterniond);

  return 0;
}