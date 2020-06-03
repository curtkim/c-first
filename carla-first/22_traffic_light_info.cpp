// from https://github.com/wx9698/carlaviz/blob/master/backend/src/backend/proxy/carla_proxy.cpp

#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <thread>
#include <chrono>
#include <ctime>
#include <memory>

#include <carla/client/ActorBlueprint.h>
#include <carla/client/BlueprintLibrary.h>
#include <carla/client/Client.h>
#include <carla/client/Map.h>
#include <carla/client/Sensor.h>
#include <carla/client/TimeoutException.h>
#include <carla/client/World.h>
#include <carla/geom/Transform.h>
#include <carla/image/ImageIO.h>
#include <carla/image/ImageView.h>
#include <carla/sensor/data/Image.h>

#include "common.hpp"
#include "xodr2geojson.hpp"
#include "utils.hpp"

namespace cc = carla::client;
namespace cg = carla::geom;
namespace csd = carla::sensor::data;

using namespace std::chrono_literals;
using namespace std::string_literals;

static auto ParseArguments(int argc, const char *argv[]) {
  assert((argc == 1u) || (argc == 3u));
  using ResultType = std::tuple<std::string, uint16_t>;
  return argc == 3u ?
      ResultType{argv[1u], std::stoi(argv[2u])} :
      ResultType{"localhost", 2000u};
}

static const std::string MAP_NAME = "/Game/Carla/Maps/Town03";


void FlatVector(std::vector<double>& v, const std::vector<double>& to_add, int neg_factor = 1) {
  v.push_back(to_add[0]);
  v.push_back(neg_factor * to_add[1]);
  v.push_back(to_add[2]);
}

std::pair<double, double> AfterRotate(double x, double y, double yaw) {
  return {std::cos(yaw) * x - std::sin(yaw) * y,
          std::sin(yaw) * x + std::cos(yaw) * y};
}

template<typename T>
string str(T begin, T end)
{
  stringstream ss;
  bool first = true;
  for (; begin != end; begin++)
  {
    if (!first)
      ss << ", ";
    ss << *begin;
    first = false;
  }
  return ss.str();
}

static void AddTrafficLightAreas(boost::shared_ptr<carla::client::Actor> actor,
                                      carla::SharedPtr<carla::client::Map> map) {

  std::unordered_map<uint32_t, std::vector<std::vector<double>>> traffic_lights_{};

  const double area_length = 2.0;
  auto id = actor->GetId();
  if (traffic_lights_.find(id) == traffic_lights_.end()) {
    traffic_lights_.insert({id, std::vector<std::vector<double>>()});
  }

  auto tl = boost::static_pointer_cast<carla::client::TrafficLight>(actor);
  auto trigger_volume = tl->GetTriggerVolume();
  auto transform = tl->GetTransform();
  transform.TransformPoint(trigger_volume.location);

  double x_off = trigger_volume.extent.x;
  double y_off = trigger_volume.extent.y;

  std::vector<point_3d_t> vertices;
  double yaw = transform.rotation.yaw / 180.0 * M_PI;
  auto location = trigger_volume.location;
  std::vector<std::pair<double, double>> offset = {
    AfterRotate(-x_off, -y_off, yaw), AfterRotate(-x_off, y_off, yaw),
    AfterRotate(x_off, y_off, yaw), AfterRotate(x_off, -y_off, yaw)};

  for (int j = 0; j < offset.size(); j++) {
    vertices.emplace_back(location.x + offset[j].first,
                          (location.y + offset[j].second), location.z);
  }

  auto central_waypoint = map->GetWaypoint(location);
  auto now_waypoint = central_waypoint;
  std::unordered_set<uint32_t> visited_points;

  while (now_waypoint != nullptr) {
    auto now_id = now_waypoint->GetId();
    auto lane_type = now_waypoint->GetType();
    if (visited_points.find(now_id) != visited_points.end()) {
      break;
    }
    visited_points.insert(now_id);
    if (lane_type != carla::road::Lane::LaneType::Driving) {
      now_waypoint = now_waypoint->GetLeft();
      continue;
    }
    auto loc = now_waypoint->GetTransform().location;
    point_3d_t p(loc.x, loc.y, loc.z);
    if (!Utils::IsWithin(p, vertices)) {
      break;
    }
    auto tmp_waypoints = now_waypoint->GetNext(area_length);
    if (tmp_waypoints.empty()) {
      std::cerr <<
        "the waypoint of the trigger volume of a traffic light is too "
        "close to the intersection, the map does not show the volumn";
    } else {
      std::vector<double> area;
      auto width = now_waypoint->GetLaneWidth();
      auto tmp_waypoint = tmp_waypoints[0];
      auto right_top_p = XodrGeojsonConverter::LateralShiftGetVector(
        tmp_waypoint->GetTransform(), width / 2.0);
      auto left_top_p = XodrGeojsonConverter::LateralShiftGetVector(
        tmp_waypoint->GetTransform(), -width / 2.0);
      auto right_down_p = XodrGeojsonConverter::LateralShiftGetVector(
        now_waypoint->GetTransform(), width / 2.0);
      auto left_down_p = XodrGeojsonConverter::LateralShiftGetVector(
        now_waypoint->GetTransform(), -width / 2.0);
      FlatVector(area, right_top_p, -1);
      FlatVector(area, right_down_p, -1);
      FlatVector(area, left_down_p, -1);
      FlatVector(area, left_top_p, -1);
      traffic_lights_[id].push_back(std::move(area));
    }
    now_waypoint = now_waypoint->GetLeft();
  }

  // now go through right points
  now_waypoint = central_waypoint->GetRight();
  visited_points.clear();
  while (now_waypoint != nullptr) {
    auto now_id = now_waypoint->GetId();
    auto lane_type = now_waypoint->GetType();
    if (visited_points.find(now_id) != visited_points.end()) {
      std::cout << "alread searched" << std::endl;
      break;
    }
    visited_points.insert(now_id);
    if (lane_type != carla::road::Lane::LaneType::Driving) {
      now_waypoint = now_waypoint->GetRight();
      continue;
    }
    auto loc = now_waypoint->GetTransform().location;
    point_3d_t p(loc.x, loc.y, loc.z);
    if (!Utils::IsWithin(p, vertices)) {
      break;
    }
    auto tmp_waypoints = now_waypoint->GetNext(area_length);
    if (tmp_waypoints.empty()) {
      std::cerr <<
        "the waypoint of the trigger volume of a traffic light is too "
        "close to the intersection, the map does not show the volumn";
    } else {
      std::vector<double> area;
      auto width = now_waypoint->GetLaneWidth();
      auto tmp_waypoint = tmp_waypoints[0];
      auto right_top_p = XodrGeojsonConverter::LateralShiftGetVector(
        tmp_waypoint->GetTransform(), width / 2.0);
      auto left_top_p = XodrGeojsonConverter::LateralShiftGetVector(
        tmp_waypoint->GetTransform(), -width / 2.0);
      auto right_down_p = XodrGeojsonConverter::LateralShiftGetVector(
        now_waypoint->GetTransform(), width / 2.0);
      auto left_down_p = XodrGeojsonConverter::LateralShiftGetVector(
        now_waypoint->GetTransform(), -width / 2.0);

      FlatVector(area, right_top_p, -1);
      FlatVector(area, right_down_p, -1);
      FlatVector(area, left_down_p, -1);
      FlatVector(area, left_top_p, -1);
      traffic_lights_[id].push_back(std::move(area));
    }
    now_waypoint = now_waypoint->GetRight();
  }

  for( const auto& n : traffic_lights_ ) {
    for (auto v : n.second)
      std::cout << "\t" << str(v.begin(), v.end()) << std::endl;
  }
}

int main(int argc, const char *argv[]) {
  try {
    std::cout << "main thread : " << std::this_thread::get_id() << std::endl;

    std::string host;
    uint16_t port;
    std::tie(host, port) = ParseArguments(argc, argv);

    std::mt19937_64 rng((std::random_device())());

    auto client = cc::Client(host, port, 1);
    client.SetTimeout(10s);

    std::cout << "Client API version : " << client.GetClientVersion() << '\n';
    std::cout << "Server API version : " << client.GetServerVersion() << '\n';

    auto world = client.GetWorld();
    if (!ends_with(MAP_NAME, world.GetMap()->GetName())) {
      std::cout << "load map " << MAP_NAME << std::endl;
      world = client.LoadWorld(MAP_NAME);
    }
    std::cout << "current map name: " << world.GetMap()->GetName() << std::endl;

    // Get a random vehicle blueprint.
    auto blueprint_library = world.GetBlueprintLibrary();
    auto vehicles = blueprint_library->Filter("vehicle");

    // Find a valid spawn point.
    auto map = world.GetMap();

    auto actor_snapshots = world.WaitForTick(2s);

    for (const auto& actor_snapshot : actor_snapshots) {
      auto actor = world.GetActor(actor_snapshot.id);
      if (actor == nullptr) {
        continue;
      }
      if (actor->GetTypeId() == "traffic.traffic_light") {
        std::cout << "traffic.traffic_light " << actor->GetId() << std::endl;
        AddTrafficLightAreas(actor, map);
      }
    }

    // Remove actors from the simulation.
    //camera->Destroy();
    //vehicle->Destroy();
    std::cout << "Actors destroyed." << std::endl;

  } catch (const cc::TimeoutException &e) {
    std::cout << '\n' << e.what() << std::endl;
    return 1;
  } catch (const std::exception &e) {
    std::cout << "\nException: " << e.what() << std::endl;
    return 2;
  }
}