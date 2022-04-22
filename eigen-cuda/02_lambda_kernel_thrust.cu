#include <Eigen/Dense>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

void ScalePoints(const float scale, thrust::device_vector<Eigen::Vector3f>& points) {
    using namespace Eigen;

    thrust::for_each(points.begin(), points.end(),
                     [=] __device__(Eigen::Vector3f &pt) {
                         pt = pt * scale;
                     });
}

int main() {
    thrust::host_vector<Eigen::Vector3f> host_points(4);

    host_points[0] = Eigen::Vector3f(-1., -1., 0.);
    host_points[1] = Eigen::Vector3f(-1., 1., 0.);
    host_points[2] = Eigen::Vector3f(1., 1., 0.);
    host_points[3] = Eigen::Vector3f(1., -1., 0.);

    thrust::device_vector<Eigen::Vector3f> points = host_points;

    ScalePoints(2.0f, points);
    host_points = points;

    for(auto pt : host_points){
        std::cout << pt << "\n";
    }

    return 0;
}