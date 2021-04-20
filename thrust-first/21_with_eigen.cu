#include <thrust/device_vector.h>
#include <iostream>
#include <eigen3/Eigen/Eigen>
#include <iostream>

static_assert(sizeof(Eigen::Vector3f) == 4*3);

using namespace Eigen;


namespace cupoch {

namespace is_eigen_matrix_detail {
    template <typename T>
    std::true_type test(const Eigen::MatrixBase<T>*);
    std::false_type test(...);
}

template <typename T>
struct is_eigen_matrix
        : public decltype(is_eigen_matrix_detail::test(std::declval<T*>()))
{};

template <typename VectorType, typename Enable = void>
struct elementwise_minimum;

template <typename VectorType, typename Enable = void>
struct elementwise_maximum;

template <typename VectorType>
struct elementwise_minimum<VectorType, typename std::enable_if<is_eigen_matrix<VectorType>::value>::type> {
    __device__ VectorType operator()(const VectorType &a, const VectorType &b) {
        return a.array().min(b.array()).matrix();
    }
};

template <typename VectorType>
struct elementwise_maximum<VectorType, typename std::enable_if<is_eigen_matrix<VectorType>::value>::type> {
    __device__ VectorType operator()(const VectorType &a, const VectorType &b) {
        return a.array().max(b.array()).matrix();
    }
};
}

namespace curt {

    template <typename VectorType>
    struct elementwise_minimum {
        __device__ VectorType operator()(const VectorType &a, const VectorType &b) {
            return a.array().min(b.array()).matrix();
        }
    };
}



int main(void)
{
    thrust::device_vector<Vector3f> points{4};
    points.push_back(Vector3f(0, 0, 0));
    points.push_back(Vector3f(10, 0, 0));
    points.push_back(Vector3f(0, 10, 0));
    points.push_back(Vector3f(0, 0, 10));

    //Vector3f a(0,0,0);
    //Vector3f b(1,0,0);
    //Vector3f c = a.array().min(b.array());
    //std::cout << c << "\n";

    Vector3f init = points[0];
    Vector3f min = thrust::reduce(points.begin()+1, points.end(), init, cupoch::elementwise_minimum<Vector3f>());
    std::cout << min << "\n";

    Vector3f max = thrust::reduce(points.begin()+1, points.end(), init, cupoch::elementwise_maximum<Vector3f>());
    std::cout << max << "\n";

    Vector3f min2 = thrust::reduce(points.begin()+1, points.end(), init, curt::elementwise_minimum<Vector3f>());
    std::cout << min2 << "\n";

    /* 왜 컴파일이 안되는지 알기 어렵다.
    auto minimum = [] __device__ (const Vector3f &a, const Vector3f &b) -> Vector3f {
        return a.array().min(b.array()).matrix();
    };
    Vector3f min3 = thrust::reduce(points.begin()+1, points.end(), init, minimum);
    std::cout << min3 << "\n";
    */

    return 0;
}