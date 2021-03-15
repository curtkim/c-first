#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;


void base() {
  int array[8];
  for(int i = 0; i < 8; ++i)
    array[i] = i;

  cout << "Column-major:\n" << Map<Matrix<int,2,4> >(array) << endl;
  cout << "Row-major:\n" << Map<Matrix<int,2,4,RowMajor> >(array) << endl;

  cout << "Row-major using stride:\n" <<
       Map<Matrix<int,2,4>, Unaligned, Stride<1,4> >(array) << endl;
}

void using_map_variable() {
  typedef Matrix<float,1,Dynamic> MatrixType;
  typedef Map<MatrixType> MapType;
  typedef Map<const MatrixType> MapTypeConst;   // a read-only map

  const int n_dims = 5;

  MatrixType m1(n_dims);
  m1.setRandom();
  cout << "m1: " << m1 << endl;

  MatrixType m2(n_dims);
  m2.setRandom();
  cout << "m2: " << m2 << endl;
  cout << "Squared euclidean distance: " << (m1-m2).squaredNorm() << endl;

  // 3. share
  float *p = &m2(0);  // get the address storing the data for m2
  MapType m2map(p,m2.size());   // m2map shares data with m2
  cout << "Squared euclidean distance, using map: " <<
       (m1-m2map).squaredNorm() << endl;

  // 4. const
  MapTypeConst m2mapconst(p, m2.size());  // a read-only accessor for m2
  m2map(3) = 7;   // this will change m2, since they share the same array
  cout << "Updated m2: " << m2 << endl;
  cout << "m2 constant accessor: " << m2mapconst(3) << endl;

  /* m2mapconst(2) = 5; */   // this yields a compile-time error
}

void change_the_mapped_array() {
  int data[] = {1,2,3,4,5,6,7,8,9};
  Map<RowVectorXi> v(data, 4);
  cout << "The mapped vector v is: " << v << "\n";

  new (&v) Map<RowVectorXi>(data+4,5);
  cout << "Now v is: " << v << "\n";
}

int main() {
  base();
  using_map_variable();
  change_the_mapped_array();
}