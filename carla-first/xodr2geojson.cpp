#include "xodr2geojson.hpp"

using point_3d_t = boost::geometry::model::point<double, 3, boost::geometry::cs::cartesian>;

std::string XodrGeojsonConverter::Convert(std::string xodr) {

  //carla::client::Map map("map", xodr);
  // why must use shared_ptr?
  auto pMap = boost::make_shared<carla::client::Map>("map", xodr);
  auto topology = pMap->GetTopology();
  auto json = InitGeoJson();
  uint32_t idx = 0u;
  for (const auto& point_pair : topology) {
    std::vector<point_3d_t> points;
    points.push_back(LateralShift(point_pair.first->GetTransform(),
                                  point_pair.first->GetLaneWidth()));
    points.push_back(LateralShift(point_pair.second->GetTransform(),
                                  point_pair.second->GetLaneWidth()));
    AddOneLine(points, point_pair.first->GetRoadId(), json, idx);
    idx++;
  }
  return json.dump();
}


std::string XodrGeojsonConverter::GetGeoJsonFromCarlaMap(
    boost::shared_ptr<carla::client::Map> map_ptr) {
  auto topology = map_ptr->GetTopology();
  auto json = InitGeoJson();
  uint32_t idx = 0u;
  for (const auto& point_pair : topology) {
    // auto waypoint = point_pair.first;
    AddOneSide(point_pair.first, json, idx);
    AddOneSide(point_pair.second, json, idx + 2);
    idx += 4;
  }
  return json.dump();
}

nlohmann::json XodrGeojsonConverter::InitGeoJson() {
  nlohmann::json json;
  json["type"] = "FeatureCollection";
  return std::move(json);
}

void XodrGeojsonConverter::AddOneSide(
    const carla::SharedPtr<carla::client::Waypoint>& waypoint,
    nlohmann::json& json, const uint32_t& index) {
  std::vector<carla::SharedPtr<carla::client::Waypoint>> tmp_waypoints;
  uint32_t road_id = waypoint->GetRoadId();
  auto next_waypoints = waypoint->GetNext(precision_);
  tmp_waypoints.push_back(waypoint);
  while (!next_waypoints.empty()) {
    auto next_waypoint = next_waypoints[0];
    if (next_waypoint->GetRoadId() == road_id) {
      tmp_waypoints.push_back(next_waypoint);
      next_waypoints = next_waypoint->GetNext(precision_);
    } else {
      break;
    }
  }
  std::vector<point_3d_t> points;
  for (const auto& waypoint : tmp_waypoints) {
    points.push_back(LateralShift(waypoint->GetTransform(),
                                  -waypoint->GetLaneWidth() * 0.5));
  }
  AddOneLine(points, road_id, json, index);
  points.clear();
  for (const auto& waypoint : tmp_waypoints) {
    points.push_back(
        LateralShift(waypoint->GetTransform(), waypoint->GetLaneWidth() * 0.5));
  }
  AddOneLine(points, road_id, json, index + 1);
}

void XodrGeojsonConverter::AddOneLine(const std::vector<point_3d_t>& points,
                                      const uint32_t& road_id,
                                      nlohmann::json& json,
                                      const uint32_t& index) {
  json["features"][index]["type"] = "Feature";
  json["features"][index]["id"] = std::to_string(index);
  json["features"][index]["properties"]["name"] = std::to_string(road_id);
  json["features"][index]["geometry"]["type"] = "LineString";
  int i = 0;
  for (const auto& point : points) {
    json["features"][index]["geometry"]["coordinates"][i][0] = point.get<0>();
    json["features"][index]["geometry"]["coordinates"][i][1] = -point.get<1>();
    json["features"][index]["geometry"]["coordinates"][i][2] = point.get<2>();
    i++;
  }
}

point_3d_t XodrGeojsonConverter::LateralShift(carla::geom::Transform transform,
                                              double shift) {
  transform.rotation.yaw += 90.0;
  point_3d_t p1(transform.location.x, transform.location.y,
                transform.location.z);
  auto p2_tmp = shift * transform.GetForwardVector();
  point_3d_t p2(p2_tmp.x, p2_tmp.y, p2_tmp.z);
  // auto point = transform.location + shift * transform.GetForwardVector();
  return point_3d_t(p1.get<0>() + p2.get<0>(), p1.get<1>() + p2.get<1>(),
                    p1.get<2>() + p2.get<2>());
}

std::vector<double> XodrGeojsonConverter::LateralShiftGetVector(carla::geom::Transform transform, double shift) {
  transform.rotation.yaw += 90.0;
  // point_3d_t p1(transform.location.x, transform.location.y,
  //               transform.location.z);
  auto p2_tmp = shift * transform.GetForwardVector();
  return {transform.location.x + p2_tmp.x, transform.location.y + p2_tmp.y, transform.location.z + p2_tmp.z};
}




