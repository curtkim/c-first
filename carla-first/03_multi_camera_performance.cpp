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
#include <atomic>

namespace cc = carla::client;
namespace cg = carla::geom;
namespace csd = carla::sensor::data;

using namespace std::chrono_literals;
using namespace std::string_literals;

static const std::string MAP_NAME = "/Game/Carla/Maps/Town03";

void test(int carla_worker_thread, std::string camera_tick, int width, int height, int camera_count) {
  std::cout
    << "carla_worker_thread: " << carla_worker_thread
    << ", camera_tick: " << camera_tick
    << ", width: " << width
    << ", height: " << height
    << ", camera_count: " << camera_count
    << std::endl;

  try {
    auto client = cc::Client("localhost", 2000, carla_worker_thread);
    client.SetTimeout(10s);

    auto world = client.GetWorld();
    if (!ends_with(MAP_NAME, world.GetMap()->GetName())) {
      std::cout << "load map " << MAP_NAME << std::endl;
      world = client.LoadWorld(MAP_NAME);
    }

    // Get a random vehicle blueprint.
    auto blueprint_library = world.GetBlueprintLibrary();
    auto vehicles = blueprint_library->Filter("vehicle");
    auto blueprint = (*vehicles)[0];

    auto map = world.GetMap();
    auto transform = carla::geom::Transform(carla::geom::Location(-36.6, -194.9, 0.27), carla::geom::Rotation(0, 1.4395, 0));

    auto actor = world.SpawnActor(blueprint, transform);
    auto vehicle = boost::static_pointer_cast<cc::Vehicle>(actor);

    cc::Vehicle::Control control;
    control.throttle = 1.0f;
    vehicle->ApplyControl(control);

    auto *camera_bp = blueprint_library->Find("sensor.camera.rgb");
    assert(camera_bp != nullptr);

    const_cast<carla::client::ActorBlueprint *>(camera_bp)->SetAttribute("sensor_tick", camera_tick);
    const_cast<carla::client::ActorBlueprint *>(camera_bp)->SetAttribute("image_size_x", std::to_string(width));
    const_cast<carla::client::ActorBlueprint *>(camera_bp)->SetAttribute("image_size_y", std::to_string(height));

    std::atomic<long> count(0);

    std::vector<boost::shared_ptr<cc::Sensor>> cameras;
    for(int i = 0; i < camera_count; i++){
      auto camera_transform = cg::Transform{
        cg::Location{-5.5f, 0.0f, 2.8f},   // x, y, z.
        cg::Rotation{-1.0f*i, 0.0f, 0.0f}}; // pitch, yaw, roll.

      auto camera = boost::static_pointer_cast<cc::Sensor>(world.SpawnActor(*camera_bp, camera_transform, vehicle.get()));
      cameras.push_back(camera);

      camera->Listen([&count](auto data) {
          count++;
      });
    }

    std::this_thread::sleep_for(10s);

    //std::cout << "count2: " << count2 << std::endl;

    // Remove actors from the simulation.
    for(auto camera : cameras)
      camera->Destroy();
    vehicle->Destroy();

    std::cout << "count: " << count << std::endl;

  } catch (const cc::TimeoutException &e) {
    std::cout << '\n' << e.what() << std::endl;
  } catch (const std::exception &e) {
    std::cout << "\nException: " << e.what() << std::endl;
  }
}

int main(int argc, const char *argv[]) {
  test(1, "0.1", 800, 600, 1);

  test(1, "0.1", 800, 600, 4);
  test(1, "0.1", 800, 600, 8);
  test(1, "0.1", 800, 600, 12);

  test(1, "0.1", 1600, 1200, 12);

  test(1, "0.05", 800, 600, 4);
  test(1, "0.05", 800, 600, 6);
  test(1, "0.05", 800, 600, 8);
  test(1, "0.05", 800, 600, 12);

  test(1, "0.033", 800, 600, 1);
  test(1, "0.033", 800, 600, 4);
  test(1, "0.033", 800, 600, 8);
  test(1, "0.033", 800, 600, 12);

  test(4, "0.033", 800, 600, 4);

  return 0;
}