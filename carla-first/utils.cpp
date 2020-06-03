#include "utils.hpp"


point_3d_t Utils::GetOffsetAfterTransform(const point_3d_t& origin,
                                          double yaw) {
  double x = origin.get<0>();
  double y = origin.get<1>();
  return point_3d_t(std::cos(yaw) * x - std::sin(yaw) * y,
                    std::sin(yaw) * x + std::cos(yaw) * y, origin.get<2>());
}

bool Utils::IsStartWith(const std::string& origin, const std::string& pattern) {
  size_t o_len = origin.size();
  size_t p_len = pattern.size();
  if (p_len <= 0u) {
    return true;
  }
  if (o_len < p_len) {
    return false;
  }
  return (origin.substr(0, p_len) == pattern);
}

bool Utils::IsWithin(const point_3d_t& point,
                     const std::vector<point_3d_t>& polygon) {
  typedef boost::geometry::model::d2::point_xy<double> point_type;
  typedef boost::geometry::model::polygon<point_type> polygon_type;
  point_type poi(point.get<0>(), point.get<1>());
  std::vector<point_type> points;
  for (const auto& point : polygon) {
    points.emplace_back(point.get<0>(), point.get<1>());
  }
  points.emplace_back(polygon[0].get<0>(), polygon[0].get<1>());
  polygon_type p;
  boost::geometry::assign_points(p, points);
  return boost::geometry::within(poi, p);
}

double Utils::ComputeSpeed(const carla::geom::Vector3D& velo) {
  double res = velo.x * velo.x + velo.y * velo.y + velo.z * velo.z;
  return std::sqrt(res);
}