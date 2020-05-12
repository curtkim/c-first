#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>

using namespace std;
using namespace Eigen;

int main() {
  float arrVertices[] = {-1.0, -1.0, -1.0,
                         1.0, -1.0, -1.0,
                         1.0, 1.0, -1.0,
                         -1.0, 1.0, -1.0,
                         -1.0, -1.0, 1.0,
                         1.0, -1.0, 1.0,
                         1.0, 1.0, 1.0,
                         -1.0, 1.0, 1.0};

  // Map
  MatrixXf mVertices = Map<Matrix<float, 3, 8> >(arrVertices);

  // U = TRSI
  Transform<float, 3, Affine> t = Transform<float, 3, Affine>::Identity();
  t.scale(0.8f);
  t.rotate(AngleAxisf(0.25f * M_PI, Vector3f::UnitX()));
  t.translate(Vector3f(1.5, 10.2, -5.1));

  cout << mVertices.rows() << "," << mVertices.cols() << endl;
  cout << mVertices << endl;

  cout << "------------------" << endl;
  cout << mVertices.colwise().homogeneous() << endl;
  cout << "------------------" << endl;

  cout << t * mVertices.colwise().homogeneous() << endl;
}