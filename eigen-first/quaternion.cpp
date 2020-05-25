#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>

using namespace std;
using namespace Eigen;

int main() {
  Eigen::Quaterniond q(2, 0, 1, -3);
  std::cout << "This quaternion consists of a scalar " << q.w() << " and a vector " << std::endl << q.vec() << std::endl;

  q.normalize();
  std::cout << "To represent rotation, we need to normalize it such that its length is " << q.norm() << std::endl;
  std::cout << "This quaternion consists of a scalar " << q.w() << " and a vector " << std::endl << q.vec() << std::endl;

  Eigen::Vector3d v(1, 2, -1);
  Eigen::Quaterniond p;
  p.w() = 0;
  p.vec() = v;

  Eigen::Quaterniond rotatedP = q * p * q.inverse();
  Eigen::Vector3d rotatedV = rotatedP.vec();
  std::cout << "We can now use it to rotate a vector " << std::endl << v << " to " << std::endl << rotatedV << std::endl;

  Eigen::Matrix3d R = q.toRotationMatrix(); // convert a quaternion to a 3x3 rotation matrix
  std::cout << "Compare with the result using an rotation matrix " << std::endl << R * v << std::endl;

  Eigen::Quaterniond a = Eigen::Quaterniond::Identity();
  Eigen::Quaterniond b = Eigen::Quaterniond::Identity();
  Eigen::Quaterniond c; // Adding two quaternion as two 4x1 vectors is not supported by the EIgen API. That is, c = a + b is not allowed. We have to do this in a hard way

  c.w() = a.w() + b.w();
  c.x() = a.x() + b.x();
  c.y() = a.y() + b.y();
  c.z() = a.z() + b.z();

}