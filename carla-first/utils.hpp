//
// Created by curt on 20. 6. 3..
//

#pragma once

#include <boost/geometry.hpp>
#include <boost/geometry/geometries/polygon.hpp>

#include <carla/geom/Vector3D.h>

using point_3d_t = boost::geometry::model::point<double, 3, boost::geometry::cs::cartesian>;


class Utils {
public:
    static point_3d_t GetOffsetAfterTransform(const point_3d_t& origin,
                                              double yaw);
    static bool IsStartWith(const std::string& origin,
                            const std::string& pattern);
    static bool IsWithin(const point_3d_t& point,
                         const std::vector<point_3d_t>& polygon);
    static double ComputeSpeed(const carla::geom::Vector3D& velo);
};
