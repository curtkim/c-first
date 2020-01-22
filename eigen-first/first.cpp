#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

void sample() {
    MatrixXd m(2,2);
    m(0,0) = 3;
    m(1,0) = 2.5;
    m(0,1) = -1;
    m(1,1) = m(1,0) + m(0,1);
    std::cout << m << std::endl;
}

int multiply()
{
    MatrixXd m = MatrixXd::Random(3,3);
    m = (m + MatrixXd::Constant(3,3,1.2)) * 50;
    cout << "m =" << endl << m << endl;
    VectorXd v(3);
    v << 1, 2, 3;
    cout << "m * v =" << endl << m * v << endl;
}

int multiply_fixed() {
    Matrix3d m = Matrix3d::Random();
    m = (m + Matrix3d::Constant(1.2)) * 50;
    cout << "m =" << endl << m << endl;
    Vector3d v(1,2,3);

    cout << "m * v =" << endl << m * v << endl;
}

int main()
{
    sample();
    multiply();
    multiply_fixed();

    Matrix <short , 2 , 2 > M1;
    M1 << 1, 2, 3, 4;
    cout << M1;

    return 0;
}
