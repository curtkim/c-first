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
#include <fstream>

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

auto prepare(cc::World world) {

  // Get a random vehicle blueprint.
  auto blueprint_library = world.GetBlueprintLibrary();
  auto vehicles = blueprint_library->Filter("vehicle");
  auto blueprint = (*vehicles)[0];

  // Find a valid spawn point.
  //auto map = world.GetMap();
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

  auto *camera_bp = blueprint_library->Find("sensor.camera.rgb");
  assert(camera_bp != nullptr);
  const_cast<carla::client::ActorBlueprint *>(camera_bp)->SetAttribute("sensor_tick", "0.033");

  /*
//  [carla.Transform(carla.Location(x=1.5, y=0, z=2.4), carla.Rotation(yaw=0)), 120, (UNIT_WIDTH, 0), None],
//  [carla.Transform(carla.Location(x=1.5, y=0.1, z=2.4), carla.Rotation(yaw=0)), 90, (2*UNIT_WIDTH, 0), None],
//  [carla.Transform(carla.Location(x=0, y=1.1, z=2.4), carla.Rotation(yaw=45)), 90, (2*UNIT_WIDTH, UNIT_HEIGHT), None],
//  [carla.Transform(carla.Location(x=1.0, y=1.1, z=1), carla.Rotation(yaw=135)), 120, (2*UNIT_WIDTH, 2*UNIT_HEIGHT), None],
//  [carla.Transform(carla.Location(x=-1.5, y=0, z=2.4), carla.Rotation(yaw=180)), 120, (UNIT_WIDTH, 2*UNIT_HEIGHT), None],
//  [carla.Transform(carla.Location(x=1.0, y=-1.1, z=1), carla.Rotation(yaw=225)), 120, (0, 2*UNIT_HEIGHT), None],
//  [carla.Transform(carla.Location(x=0, y=-1.1, z=2.4), carla.Rotation(yaw=315)), 90, (0, UNIT_HEIGHT), None],
//  [carla.Transform(carla.Location(x=1.5, y=-0.1, z=2.4), carla.Rotation(yaw=0)), 45, (0, 0), None],

  auto tfs = std::vector<cg::Transform>{
    cg::Transform{cg::Location{1.5f, 0.0f, 2.4f}, cg::Rotation{0.0f, 0.0f, 0.0f}},
    cg::Transform{cg::Location{1.5f, 0.0f, 2.4f}, cg::Rotation{0.0f, 0.0f, 0.0f}},
    cg::Transform{cg::Location{0.0f, 1.1f, 2.4f}, cg::Rotation{0.0f, 45.0f, 0.0f}},
    cg::Transform{cg::Location{1.0f, 1.1f, 1.0f}, cg::Rotation{0.0f, 135.0f, 0.0f}},
    cg::Transform{cg::Location{-1.5f, 0.0f, 2.4f}, cg::Rotation{0.0f, 180.0f, 0.0f}},
    cg::Transform{cg::Location{1.0f, -1.1f, 1.0f}, cg::Rotation{0.0f, 225.0f, 0.0f}},
    cg::Transform{cg::Location{0.0f, -1.1f, 2.4f}, cg::Rotation{0.0f, 315.0f, 0.0f}},
    cg::Transform{cg::Location{1.5f, -0.1f, 2.4f}, cg::Rotation{0.0f, 0.0f, 0.0f}},
  };
  // Spawn a camera attached to the vehicle.

  std::vector<rxcpp:observable<boost::shared_ptr<csd::Image>>> image_list;
  for(auto tf : tfs){
    auto cam_actor = world.SpawnActor(*camera_bp, tf, actor.get());
    auto camera = boost::static_pointer_cast<cc::Sensor>(cam_actor);
    auto image$ = from_sensor2<csd::Image>(camera);
    image$_list.emplace_back(image$);
  }
  */

  // Spawn a camera attached to the vehicle.
  auto camera_transform = cg::Transform{
    cg::Location{-5.5f, 0.0f, 2.8f},   // x, y, z.
    cg::Rotation{-15.0f, 0.0f, 0.0f}}; // pitch, yaw, roll.
  auto cam_actor = world.SpawnActor(*camera_bp, camera_transform, actor.get());
  auto camera = boost::static_pointer_cast<cc::Sensor>(cam_actor);
  auto image$ = from_sensor2<csd::Image>(camera);

  auto camera_transform2 = cg::Transform{
    cg::Location{-0.0f, 0.0f, 2.8f},   // x, y, z.
    cg::Rotation{0.0f, 0.0f, 0.0f}}; // pitch, yaw, roll.
  auto cam_actor2 = world.SpawnActor(*camera_bp, camera_transform2, actor.get());
  auto camera2 = boost::static_pointer_cast<cc::Sensor>(cam_actor2);
  auto image2$ = from_sensor2<csd::Image>(camera2);

  auto camera_transform3 = cg::Transform{
    cg::Location{-0.0f, 0.0f, 2.8f},   // x, y, z.
    cg::Rotation{0.0f, 0.0f, 0.0f}}; // pitch, yaw, roll.
  auto cam_actor3 = world.SpawnActor(*camera_bp, camera_transform3, actor.get());
  auto camera3 = boost::static_pointer_cast<cc::Sensor>(cam_actor3);
  auto image3$ = from_sensor2<csd::Image>(camera3);


  return std::make_tuple(vehicle, camera, image$, camera2, image2$, camera3, image3$);
}

int main(int argc, const char *argv[]) {

  std::cout << "main thread : " << std::this_thread::get_id() << std::endl;

  auto client = cc::Client("localhost", 2000, 3);
  client.SetTimeout(10s);
  std::cout << "Client API version : " << client.GetClientVersion() << '\n';
  std::cout << "Server API version : " << client.GetServerVersion() << '\n';

  auto world = client.GetWorld();
  if (!ends_with(MAP_NAME, world.GetMap()->GetName())) {
    std::cout << "load map " << MAP_NAME << std::endl;
    world = client.LoadWorld(MAP_NAME);
  }
  std::cout << "current map name: " << world.GetMap()->GetName() << std::endl;

  auto [vehicle, camera, image$, camera2, image2$, camera3, image3$] = prepare(world);

  std::ofstream myfile ("/data/dump.om", std::ios::out | std::ios::binary); // std::ios::app |

  //auto threads = rxcpp::observe_on_new_thread();
  auto scheduler = rxcpp::observe_on_event_loop();
  auto trigger = rxcpp::observable<>::timer(std::chrono::seconds(10));

  int count = 0;
  image$.merge(image2$, image3$)
    //.tap([](boost::shared_ptr<csd::Image> v){
    //  std::cout << std::this_thread::get_id() << " tap " << v->GetFrame() << " " << getEpochMillisecond() << std::endl;
    //})
    .observe_on(scheduler)
    .take_until(trigger)
    .as_blocking()
    .subscribe(
      [&myfile, &count](boost::shared_ptr<csd::Image> v){
        //std::cout << std::this_thread::get_id() << " onNext " << v->GetFrame() << " " << getEpochMillisecond() << std::endl;
        //myfile.write( (char*)(v->data()), v->size() );
        //myfile.flush();
        count++;
      },
      [](){
        std::cout << std::this_thread::get_id() << " OnCompleted" << std::endl;
      }
    );

  myfile.close();
  std::cout << count << std::endl;

  // Remove actors from the simulation.
  camera->Destroy();
  camera2->Destroy();
  camera3->Destroy();
  vehicle->Destroy();
  std::cout << std::this_thread::get_id() << " Actors destroyed." << std::endl;

  return 0;
}