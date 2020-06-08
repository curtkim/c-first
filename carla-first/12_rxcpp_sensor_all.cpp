#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <chrono>
#include <ctime>

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
#include <carla/sensor/data/GnssMeasurement.h>

#include <rxcpp/rx.hpp>

#include "common.hpp"
#include "carla_common.hpp"


namespace cc = carla::client;
namespace cg = carla::geom;
namespace cs = carla::sensor;
namespace csd = carla::sensor::data;

using namespace std::chrono_literals;
using namespace std::string_literals;

long getEpochMillisecond() {
  using namespace std::chrono;
  return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}


static const std::string MAP_NAME = "/Game/Carla/Maps/Town03";

int main(int argc, const char *argv[]) {
  try {

    std::cout << "main thread : " << std::this_thread::get_id() << std::endl;

    auto client = cc::Client("localhost", 2000, 10);
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
    auto blueprint = (*vehicles)[0];

    // Find a valid spawn point.
    auto map = world.GetMap();
    auto transform = carla::geom::Transform(
        carla::geom::Location(-36.6, -194.9, 0.27),
        carla::geom::Rotation(0, 1.4395, 0)
        );

    // Spawn the vehicle.
    auto actor = world.SpawnActor(blueprint, transform);
    std::cout << "Spawned " << actor->GetDisplayId() << '\n';
    auto vehicle = boost::static_pointer_cast<cc::Vehicle>(actor);

    // Apply control to vehicle.
    cc::Vehicle::Control control;
    control.throttle = 1.0f;
    vehicle->ApplyControl(control);

    // Sensor
    auto camera_transform = cg::Transform{
        cg::Location{-5.5f, 0.0f, 2.8f},   // x, y, z.
        cg::Rotation{-15.0f, 0.0f, 0.0f}}; // pitch, yaw, roll.
    std::map<std::string, std::string> caemra_attributes = {{"sensor_tick", "0.033"}};
    auto [camera, image$] = from_sensor_data<csd::Image>(
        world, "sensor.camera.rgb", caemra_attributes, camera_transform,
        vehicle);

    image$
        .subscribe(
            [](auto v){
              std::cout << std::this_thread::get_id() << " " << getEpochMillisecond() <<" camera onNext " << v->GetFrame() << std::endl;
            },
            [](){
              std::cout << std::this_thread::get_id() << " camera OnCompleted" << std::endl;
            }
        );

    auto camera_transform2 = cg::Transform{
      cg::Location{-5.5f, 0.0f, 2.8f},   // x, y, z.
      cg::Rotation{-15.0f, 0.0f, 0.0f}}; // pitch, yaw, roll.
    std::map<std::string, std::string> caemra_attributes2 = {{"sensor_tick", "0.033"}};
    auto [camera2, image2$] = from_sensor_data<csd::Image>(
      world, "sensor.camera.rgb", caemra_attributes2, camera_transform2,
      vehicle);

    image2$
      .subscribe(
        [](auto v){
            std::cout << std::this_thread::get_id() << " " << getEpochMillisecond() << " camera2 onNext " << v->GetFrame() << std::endl;
        },
        [](){
            std::cout << std::this_thread::get_id() << " camera2 OnCompleted" << std::endl;
        }
      );

    auto lidar_transform = cg::Transform{
        cg::Location{-5.5f, 0.0f, 2.8f},   // x, y, z.
        cg::Rotation{-15.0f, 0.0f, 0.0f}}; // pitch, yaw, roll.
    std::map<std::string, std::string> lidar_attributes = {{"sensor_tick", "0.1"}};
    auto [lidar, lidar$] = from_sensor_data<csd::LidarMeasurement>(
        world, "sensor.lidar.ray_cast", lidar_attributes, lidar_transform,
        vehicle);
    lidar$
        .subscribe(
            [](auto v){
              std::cout << std::this_thread::get_id() << " " << getEpochMillisecond() << " lidar onNext " << v->GetFrame() << std::endl;
            },
            [](){
              std::cout << std::this_thread::get_id() << " lidar OnCompleted" << std::endl;
            }
        );

    auto gnss_transform = cg::Transform{
        cg::Location{0.0f, 0.0f, 0.0f}, // x, y, z.
        cg::Rotation{0.0f, 0.0f, 0.0f}}; // pitch, yaw, roll.
    std::map<std::string, std::string> gnss_attributes = {};
    auto [gnss, gnss$] = from_sensor_data<csd::LidarMeasurement>(
        world, "sensor.other.gnss", gnss_attributes, gnss_transform, vehicle);
    gnss$
        .subscribe(
            [](auto v){
              std::cout << std::this_thread::get_id() << " " << getEpochMillisecond() << " gnss onNext " << v->GetFrame() << std::endl;
            },
            [](){
              std::cout << std::this_thread::get_id() << " gnss OnCompleted" << std::endl;
            }
        );

    auto imu_transform = cg::Transform{
        cg::Location{0.0f, 0.0f, 0.0f}, // x, y, z.
        cg::Rotation{0.0f, 0.0f, 0.0f}}; // pitch, yaw, roll.
    std::map<std::string, std::string> imu_attributes = {};
    auto [imu, imu$] = from_sensor_data<csd::LidarMeasurement>(
        world, "sensor.other.imu", imu_attributes, imu_transform, vehicle);
    imu$
        .subscribe(
            [](auto v){
              std::cout << std::this_thread::get_id() << " " << getEpochMillisecond() << " imu onNext " << v->GetFrame() << std::endl;
            },
            [](){
              std::cout << std::this_thread::get_id() << " imu OnCompleted" << std::endl;
            }
        );

    std::cout << std::this_thread::get_id() << " sleep" << std::endl;
    std::this_thread::sleep_for(5s);


    // Remove actors from the simulation.
    imu->Destroy();
    gnss->Destroy();
    lidar->Destroy();
    camera->Destroy();
    camera2->Destroy();
    vehicle->Destroy();
    std::cout << std::this_thread::get_id() << " Actors destroyed." << std::endl;

  } catch (const cc::TimeoutException &e) {
    std::cout << '\n' << e.what() << std::endl;
    return 1;
  } catch (const std::exception &e) {
    std::cout << "\nException: " << e.what() << std::endl;
    return 2;
  }
}

