#include <vector>
#include <Eigen/Core>
#include <iostream>

#include "03_kernel.hpp"


int main(){
  std::vector<Eigen::Vector3d> v1(10*1024, Eigen::Vector3d{ 1.0, 1.0, 1.0 });
  std::vector<Eigen::Vector3d> v2(10*1024, Eigen::Vector3d{ -1.0, 1.0, 1.0 });

  double x = Kernel::dot(v1,v2);
  std::cout << x << "\n";

  return 0;
}