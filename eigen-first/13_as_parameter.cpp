// https://eigen.tuxfamily.org/dox/TopicFunctionTakingEigenTypes.html
#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;


template <typename Derived>
void print_size(const EigenBase<Derived>& b)
{
  std::cout << "size (rows, cols): " << b.size() << " (" << b.rows()
            << ", " << b.cols() << ")" << std::endl;
}

void eigen_base() {
  Vector3f v;
  print_size(v);
  // v.asDiagonal() returns a 3x3 diagonal matrix pseudo-expression
  print_size(v.asDiagonal());
}

template <typename Derived>
void print_block(const DenseBase<Derived>& b, int x, int y, int r, int c)
{
  std::cout << "block: " << b.block(x,y,r,c) << std::endl;
}

template <typename Derived>
void print_max_coeff(const ArrayBase<Derived> &a)
{
  std::cout << "max: " << a.maxCoeff() << std::endl;
}


int main() {
  eigen_base();
}