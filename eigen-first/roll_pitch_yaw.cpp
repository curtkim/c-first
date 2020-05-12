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

int main() {
  Eigen::Quaterniond quaterniond = euler2Quaternion(0, 0, M_PI);
  std::cout << quaterniond.matrix() << std::endl;
  
  //Eigen::Matrix3d rot_mat = rotation_from_euler(0, 0, 0.5*M_PI);
  //std::cout << rot_mat << std::endl;
  return 0;
}