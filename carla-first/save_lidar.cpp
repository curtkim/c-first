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
#include <carla/sensor/data/LidarMeasurement.h>
#include <carla/pointcloud/PointCloudIO.h>

#include "common.hpp"


namespace cc = carla::client;
namespace cg = carla::geom;
namespace csd = carla::sensor::data;

using namespace std::chrono_literals;
using namespace std::string_literals;

#define EXPECT_TRUE(pred) if (!(pred)) { throw std::runtime_error(#pred); }

/// Pick a random element from @a range.
template <typename RangeT, typename RNG>
static auto &RandomChoice(const RangeT &range, RNG &&generator) {
  EXPECT_TRUE(range.size() > 0u);
  std::uniform_int_distribution<size_t> dist{0u, range.size() - 1u};
  return range[dist(std::forward<RNG>(generator))];
}


static auto ParseArguments(int argc, const char *argv[]) {
  EXPECT_TRUE((argc == 1u) || (argc == 3u));
  using ResultType = std::tuple<std::string, uint16_t>;
  return argc == 3u ?
         ResultType{argv[1u], std::stoi(argv[2u])} :
         ResultType{"localhost", 2000u};
}


static const std::string MAP_NAME = "/Game/Carla/Maps/Town03";

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

    // Load a random town.
    // auto town_name = RandomChoice(client.GetAvailableMaps(), rng);
    // std::cout << "Loading world: " << town_name << std::endl;
    // auto world = client.LoadWorld("/Game/Carla/Maps/Town04");
    auto world = client.GetWorld();
    if (!ends_with(MAP_NAME, world.GetMap()->GetName())) {
      std::cout << "load map " << MAP_NAME << std::endl;
      world = client.LoadWorld(MAP_NAME);
    }
    std::cout << "current map name: " << world.GetMap()->GetName() << std::endl;

    // Get a random vehicle blueprint.
    auto blueprint_library = world.GetBlueprintLibrary();
    auto vehicles = blueprint_library->Filter("vehicle");
    auto blueprint = RandomChoice(*vehicles, rng);

    // Find a valid spawn point.
    auto map = world.GetMap();
    //auto transform = RandomChoice(map->GetRecommendedSpawnPoints(), rng);
    auto transform = carla::geom::Transform(carla::geom::Location(-36.6, -194.9, 0.27), carla::geom::Rotation(0, 1.4395, 0));

    // Spawn the vehicle.
    auto actor = world.SpawnActor(blueprint, transform);
    std::cout << "Spawned " << actor->GetDisplayId() << '\n';
    auto vehicle = boost::static_pointer_cast<cc::Vehicle>(actor);

    // Apply control to vehicle.
    cc::Vehicle::Control control;
    control.throttle = 1.0f;
    vehicle->ApplyControl(control);

    // Move spectator so we can see the vehicle from the simulator window.
    auto spectator = world.GetSpectator();
    transform.location += 32.0f * transform.GetForwardVector();
    transform.location.z += 2.0f;
    transform.rotation.yaw += 180.0f;
    transform.rotation.pitch = -15.0f;
    spectator->SetTransform(transform);


    // Find a camera blueprint.
    auto *lidar_bp = blueprint_library->Find("sensor.lidar.ray_cast");
    EXPECT_TRUE(lidar_bp != nullptr);
    //const_cast<carla::client::ActorBlueprint *>(camera_bp)->SetAttribute("sensor_tick", "0.033");


    // Spawn a camera attached to the vehicle.
    auto lidar_transform = cg::Transform{
        cg::Location{-5.5f, 0.0f, 2.8f},   // x, y, z.
        cg::Rotation{-15.0f, 0.0f, 0.0f}}; // pitch, yaw, roll.
    auto lidar_actor = world.SpawnActor(*lidar_bp, lidar_transform, actor.get());
    auto lidar = boost::static_pointer_cast<cc::Sensor>(lidar_actor);

    // Register a callback to save images to disk.
    lidar->Listen([](auto data) {
      auto lidar_data = boost::static_pointer_cast<csd::LidarMeasurement>(data);
      EXPECT_TRUE(lidar_data != nullptr);

      std::cout << std::this_thread::get_id() << " " << lidar_data->GetFrame() << " ch_count:" << lidar_data->GetChannelCount()
          << " pt_count(0):" << lidar_data->GetPointCount(0)
          << " pt_count(10):" << lidar_data->GetPointCount(10)
          << " pt_count(20):" << lidar_data->GetPointCount(20)
          << std::endl;

      char buffer[9u];
      std::snprintf(buffer, sizeof(buffer), "%08zu", lidar_data->GetFrame());
      auto filename = "_lidar/"s + buffer + ".ply";

      /*
      auto begin = lidar_data->begin();
      auto end = lidar_data->end();
      for (; begin != end; ++begin)
        std::cout << begin->x << ' ' << begin->y << ' ' << begin->z << '\n';
      */

      carla::pointcloud::PointCloudIO::SaveToDisk(filename, lidar_data->begin(), lidar_data->end());
    });

    std::this_thread::sleep_for(5s);

    // Remove actors from the simulation.
    lidar->Destroy();
    vehicle->Destroy();
    std::cout << "Actors destroyed." << std::endl;

  } catch (const cc::TimeoutException &e) {
    std::cout << '\n' << e.what() << std::endl;
    return 1;
  } catch (const std::exception &e) {
    std::cout << "\nException: " << e.what() << std::endl;
    return 2;
  }
}
