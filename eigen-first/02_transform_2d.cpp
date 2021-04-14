#include <Eigen/Core>
#include <Eigen/Geometry>
#include <iostream>

using namespace std;
using namespace Eigen;

int main() {
  float arrVertices[] = {-1.0, -1.0,
                         1.0, -1.0,
                         1.0, 1.0,
                         -1.0, 1.0};

  // Map
  MatrixXf mVertices = Map<Matrix<float, 2, 4> >(arrVertices);

  // U = TRSI
  Transform<float, 2, Affine> t = Transform<float, 2, Affine>::Identity();
  //t.scale(0.8f);
  //t.rotate(AngleAxisf(0.25f * M_PI, Vector3f::UnitX()));
  //t.translate(Vector2f(2, 0));
  t.rotate(0.25f * M_PI);

  cout << mVertices.rows() << "," << mVertices.cols() << endl;
  cout << mVertices << endl;

  cout << "------------------" << endl;
  cout << mVertices.colwise().homogeneous() << endl;
  cout << "------------------" << endl;

  cout << t * mVertices.colwise().homogeneous() << endl;
  MatrixXf transformed = t * mVertices.colwise().homogeneous();
  cout << transformed.rows() << " " << transformed.cols() << endl;
}