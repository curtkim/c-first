#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

int main()
{
  ArrayXf v = ArrayXf::LinSpaced(11, 0.f, 10.f); // 0~10
  cout << "size=" << v.size() << endl;

  float *vc = v.data();
  cout << vc[3] << endl;  // 3.0

  return 0;
}
