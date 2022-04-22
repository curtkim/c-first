#pragma once

#include <vector>
#include <Eigen/Core>

namespace Kernel {
    double dot(const std::vector<Eigen::Vector3d> & v1, const std::vector<Eigen::Vector3d> & v2);
}

