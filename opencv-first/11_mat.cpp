#include <stdio.h>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

void create() {
  //  we use unsigned char types that are 8 bit long and each pixel has three of these to form the three channels
  // CV_[The number of bits per item][Signed or Unsigned][Type Prefix]C[The channel number]
  Mat A(2,2, CV_8UC3, Scalar(0,0,255));
  cout << "A = " << endl << " " << A << endl;

  Mat B(A);
  cout << "B = " << endl << " " << B << endl;
  printf("%ld %ld\n", A.data, B.data); // shallow copy


  int sz[2] = {2,2};
  Mat C(2, sz, CV_8UC(1), Scalar::all(0));
  cout << "C = " << endl << " " << C << endl;

  Mat D = (Mat_<double>({0, -1, 0,
                         -1, 5, -1,
                         0, -1, 0})).reshape(3);
  cout << "D = " << endl << " " << D << endl << endl;
}

void subset() {
  cout << "==============" << endl;

  Mat C(256,256, CV_8UC3, Scalar(0,0,255));

  // subset
  Mat D (C, Rect(10, 10, 100, 100) ); // using a rectangle
  cout << "D " << D.size << endl;

  // subset
  Mat E = C(Range::all(), Range(1,3)); // using row and column boundaries
  cout << "E " << E.size << endl;

}

int main(int argc, char** argv )
{
  create();
  subset();

  return 0;
}

