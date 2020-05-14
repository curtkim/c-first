#pragma once

#include <string>

#include "carla/client/Map.h"
#include "carla/client/Sensor.h"
#include "carla/client/Waypoint.h"
#include "carla/geom/Location.h"
#include "carla/geom/Transform.h"
#include "carla/opendrive/OpenDriveParser.h"
#include "carla/road/element/RoadInfoGeometry.h"
#include "carla/sensor/SensorData.h"

#include <boost/shared_ptr.hpp>
#include <boost/geometry.hpp>
#include <boost/geometry/geometries/polygon.hpp>

#include <nlohmann/json.hpp>

using namespace std;
using namespace nlohmann;
using point_3d_t = boost::geometry::model::point<double, 3, boost::geometry::cs::cartesian>;

class XodrGeojsonConverter {
public:
  static string Convert(std::string xodr);
  static string GetGeoJsonFromCarlaMap(boost::shared_ptr<carla::client::Map> map_ptr);
  static point_3d_t LateralShift(carla::geom::Transform transform, double shift);
  static vector<double> LateralShiftGetVector(carla::geom::Transform transform, double shift);

private:
  static json InitGeoJson();
  static void AddOneLine(const vector<point_3d_t>& points, const uint32_t& road_id, json& json, const uint32_t& index);
  static void AddOneSide(const carla::SharedPtr<carla::client::Waypoint>& waypoint, json& json, const uint32_t& index);
  constexpr static const double precision_{0.5};
};

