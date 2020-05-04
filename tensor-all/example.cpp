// https://eigen.tuxfamily.org/dox/group__TutorialMapClass.html
#include <iostream>

#include <Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>

using namespace std;
using namespace Eigen;

void eigen_map() {
  int array[8];
  for(int i = 0; i < 8; ++i)
    array[i] = i;

  cout << "Column-major:\n";
  cout << Map<Matrix<int,2,4> >(array) << std::endl;
  cout << "Row-major:\n";
  cout << Map<Matrix<int,2,4,RowMajor> >(array) << std::endl;
  cout << "Row-major using stride:\n";
  cout << Map<Matrix<int,2,4>, Unaligned, Stride<1,4> >(array) << std::endl;
}

void eigen_use_map() {
  typedef Matrix<float,1,Dynamic> MatrixType;
  typedef Map<MatrixType> MapType;
  typedef Map<const MatrixType> MapTypeConst;   // a read-only map
  const int n_dims = 5;

  MatrixType m1(n_dims), m2(n_dims);
  m1.setRandom();
  m2.setRandom();

  float *p = &m2(0);  // get the address storing the data for m2
  MapType m2map(p,m2.size());   // m2map shares data with m2
  MapTypeConst m2mapconst(p,m2.size());  // a read-only accessor for m2

  cout << "m1: " << m1 << endl;
  cout << "m2: " << m2 << endl;
  cout << "Squared euclidean distance: " << (m1-m2).squaredNorm() << endl;
  cout << "Squared euclidean distance, using map: " << (m1-m2map).squaredNorm() << endl;

  m2map(2) = 7;   // this will change m2, since they share the same array
  cout << "Updated m2: " << m2 << endl;
  cout << "m2 coefficient 2, constant accessor: " << m2mapconst(2) << endl;

  /* m2mapconst(2) = 5; */   // this yields a compile-time error
}

void change_mapped_array() {
  int data[] = {1,2,3,4,5,6,7,8,9};
  Map<RowVectorXi> v(data,4);
  cout << "The mapped vector v is: " << v << "\n";

  new (&v) Map<RowVectorXi>(data+4,5);
  cout << "Now v is: " << v << "\n";
}

void eigen2cv_copy() {
  Eigen::MatrixXf E = Eigen::MatrixXf::Random(3, 3);
  std::cout << "EigenMat:\n" << E << std::endl;

  cv::Mat C;
  cv::eigen2cv(E, C);
  std::cout << "cvMat:\n" << C << std::endl;

  C.at<float>(0,0) = 0;
  std::cout << "cvMat:\n" << C << std::endl;
  std::cout << "EigenMat:\n" << E << std::endl;
}

int main(int argc, char* argv[])
{
  eigen_map();
  cout << "-----------" << endl;
  eigen_use_map();

  cout << "-----------" << endl;
  change_mapped_array();

  cout << "-----------" << endl;
  eigen2cv_copy();

  return 0;
}

