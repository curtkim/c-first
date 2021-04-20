#include <Eigen/Dense>
#include <type_traits>

using namespace Eigen;
using namespace std;

static_assert(std::is_same_v<Eigen::Vector3f, Matrix<float, 3, 1>>);

int main() {}