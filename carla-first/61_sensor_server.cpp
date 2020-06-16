#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <tuple>
#include <fstream>

#include <rxcpp/rx.hpp>

#include "common.hpp"
#include "carla_common.hpp"

namespace cc = carla::client;
namespace cg = carla::geom;
namespace cs = carla::sensor;
namespace csd = carla::sensor::data;

static const std::string MAP_NAME = "/Game/Carla/Maps/Town03";

int main(int argc, const char *argv[]) {

  std::cout << "main thread: " << std::this_thread::get_id() << std::endl;

  auto[world, vehicle] = init_carla(MAP_NAME);
  auto camera_transform = cg::Transform{
    cg::Location{-5.5f, 0.0f, 2.8f},   // x, y, z.
    cg::Rotation{-15.0f, 0.0f, 0.0f}}; // pitch, yaw, roll.
  std::map<std::string, std::string> caemra_attributes = {{"sensor_tick", "0.033"}};
  auto[camera, image$] = from_sensor_data<csd::Image>(
    world, "sensor.camera.rgb", caemra_attributes, camera_transform,
    vehicle);

  // Apply control to vehicle.
  cc::Vehicle::Control control;
  control.throttle = 1.0f;
  vehicle->ApplyControl(control);

  std::ofstream myfile ("/data/dump.om", std::ios::out | std::ios::binary); // std::ios::app |

  auto trigger = rxcpp::observable<>::timer(std::chrono::seconds(10));

  image$
    .take_until(trigger)
    .as_blocking()
    .subscribe(
      [&myfile](boost::shared_ptr<csd::Image> v){
          std::cout << std::this_thread::get_id() << " onNext " << v->GetFrame() << " " << getEpochMillisecond() << " bytes=" << v->size() << std::endl;
          myfile.write( (char*)(v->data()), v->size() );
          myfile.flush();
      },
      [](){
          std::cout << std::this_thread::get_id() << " OnCompleted" << std::endl;
      }
    );


  // Remove actors from the simulation.
  camera->Destroy();
  vehicle->Destroy();
  std::cout << std::this_thread::get_id() << " Actors destroyed." << std::endl;
}