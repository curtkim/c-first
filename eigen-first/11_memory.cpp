#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;


void static_size() {
  cout << "--- " << __FUNCTION__ << "\n";

  Matrix<short, 2, 2> M1;
  M1 << 1, 2, 3, 4;
  cout << M1 << endl;
}

void dynamic_rows() {
  cout << "--- " << __FUNCTION__ << "\n";

  int rows = 4;
  using PointCloudXYZ = Matrix<float, Dynamic, 3>;
  PointCloudXYZ pc;
  pc.resize(rows, 3); // cols이 3이 아니면 에러가 발생한다. malloc(48) hooked
  pc << 1, 2, 3,
    1,2,4,
    1,2,5,
    1,2,6;
  cout << pc << "\n";
}

// malloc을 하지 않는다.
void matrix_3d(){
  cout << "--- " << __FUNCTION__ << "\n";
  Matrix3d m = Matrix3d::Random();
  m = (m + Matrix3d::Constant(1.2)) * 50;
  cout << "m =" << endl << m << endl;
  Vector3d v(1,2,3);

  cout << "m * v =" << endl << m * v << endl;
}

void matrix_xd() {
  cout << "--- " << __FUNCTION__ << "\n";

  MatrixXd m = MatrixXd::Random(3,3);           // malloc(8*9) hooked
  m = (m + MatrixXd::Constant(3,3,1.2)) * 50;
  cout << "m =" << endl << m << endl;
  VectorXd v(3);                                // malloc(8*3) hooked
  v << 1, 2, 3;
  cout << "m * v =" << endl << m * v << endl;   // malloc(8*3) hooked
}

int main() {
  static_size();
  static_size();
  dynamic_rows();

  int length = 10;
  int* array = new int[length]; // malloc(40) hooked
  delete[] array;

  matrix_3d();
  matrix_xd();
}