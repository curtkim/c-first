#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

void sample() {
  // heap을 사용하나?
  std::cout << "sample" << std::endl;
  MatrixXd m(2, 2);
  m(0, 0) = 3;
  m(1, 0) = 2.5;
  m(0, 1) = -1;
  m(1, 1) = m(1, 0) + m(0, 1);
  std::cout << m << std::endl;

  double* p = m.data();
  cout << p[1] << endl; // column major
}

int multiply() {
  MatrixXd m = MatrixXd::Random(3, 3);
  m = (m + MatrixXd::Constant(3, 3, 1.2)) * 50;
  cout << "m =" << endl << m << endl;
  VectorXd v(3);
  v << 1, 2, 3;
  cout << "m * v =" << endl << m * v << endl;
}

int multiply_fixed() {
  Matrix3d m = Matrix3d::Random();
  m = (m + Matrix3d::Constant(1.2)) * 50;
  cout << "m =" << endl << m << endl;
  Vector3d v(1, 2, 3);

  cout << "m * v =" << endl << m * v << endl;
}

int main() {
  sample();
  multiply();
  multiply_fixed();


  // static?
  Matrix<short, 2, 2> M1;
  M1 << 1, 2, 3, 4;
  cout << M1 << endl;

  // dynamic row?
  using PointCloudXYZ = Matrix<float, Dynamic, 3>;
  PointCloudXYZ pc;
  pc.resize(4,3); // cols이 3이 아니면 에러가 발생한다. 
  pc << 1, 2, 3,
    1,2,4,
    1,2,5,
    1,2,6;

  cout << pc << endl;

  return 0;
}
