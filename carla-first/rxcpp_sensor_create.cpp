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

#include <rxcpp/rx.hpp>

#include "common.hpp"


namespace cc = carla::client;
namespace cg = carla::geom;
namespace cs = carla::sensor;
namespace csd = carla::sensor::data;

using namespace std::chrono_literals;
using namespace std::string_literals;

static void SaveImageToDisk(const csd::Image &image) {
  using namespace carla::image;

  char buffer[9u];
  std::snprintf(buffer, sizeof(buffer), "%08zu", image.GetFrame());
  auto filename = "_images/"s + buffer + ".png";

  std::time_t t = std::time(nullptr);
  std::tm tm = *std::localtime(&t);
  std::cout << std::put_time(&tm, "%F %T") << " " << std::this_thread::get_id() << " frame: " << image.GetFrame();
  std::cout << filename << " size=" << image.size() << std::endl;

  //bgra8c_pixel_t
  auto data = image.data();

  auto view = ImageView::MakeView(image);
  ImageIO::WriteView(filename, view);
}

auto from_sensor(boost::shared_ptr<cc::Sensor> pSensor) {
  auto data$ = rxcpp::sources::create<boost::shared_ptr<cs::SensorData>>(
      [pSensor](rxcpp::subscriber<boost::shared_ptr<cs::SensorData>> s){
        std::cout << std::this_thread::get_id() << " before listen " << std::endl;
        pSensor->Listen([s](auto data){
          assert(data != nullptr);
          //boost::shared_ptr<csd::Image> image = boost::static_pointer_cast<csd::Image>(data);
          //std::cout << std::this_thread::get_id() << " in callback " << image->GetFrame() << std::endl;
          s.on_next(data);
        });
        //s.on_completed();
      });;//.subscribe_on(rxcpp::synchronize_new_thread());
  return data$;
}


static const std::string MAP_NAME = "/Game/Carla/Maps/Town03";

int main(int argc, const char *argv[]) {
  try {

    std::cout << "main thread : " << std::this_thread::get_id() << std::endl;

    auto client = cc::Client("localhost", 2000, 1);
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

    // Spawn a camera attached to the vehicle.
    auto camera_transform = cg::Transform{
      cg::Location{-5.5f, 0.0f, 2.8f},   // x, y, z.
      cg::Rotation{-15.0f, 0.0f, 0.0f}}; // pitch, yaw, roll.
    auto cam_actor = world.SpawnActor(*camera_bp, camera_transform, actor.get());
    auto camera = boost::static_pointer_cast<cc::Sensor>(cam_actor);


    // Register a callback to save images to disk.
    /*
    camera->Listen([](auto data) {
        auto image = boost::static_pointer_cast<csd::Image>(data);
        assert(image != nullptr);
        SaveImageToDisk(*image);
    });
    */

    //typedef boost::shared_ptr<csd::Image> imagePtr;

    /*
    auto image$ = rxcpp::sources::create<boost::shared_ptr<csd::Image>>(
      [&camera](rxcpp::subscriber<boost::shared_ptr<csd::Image>> s){
        std::cout << std::this_thread::get_id() << " before listen " << std::endl;

        camera->Listen([s](auto data){
          boost::shared_ptr<csd::Image> image = boost::static_pointer_cast<csd::Image>(data);
          assert(image != nullptr);
          std::cout << std::this_thread::get_id() << " in callback " << image->GetFrame() << std::endl;
          s.on_next(image);
        });
        //s.on_completed();
      });;//.subscribe_on(rxcpp::synchronize_new_thread());
    */
    auto image$ = from_sensor(camera).map([](boost::shared_ptr<cs::SensorData> data){
      return boost::static_pointer_cast<csd::Image>(data);
    });

    image$
      .map([](auto v){
        return rxcpp::sources::just(v)
                .tap([](boost::shared_ptr<csd::Image> v){
                  SaveImageToDisk(*v);
                })
                .subscribe_on(rxcpp::observe_on_new_thread());
        })
      .flat_map([](auto observable) { return observable; })
      .subscribe(
        [](auto v){
          std::cout << std::this_thread::get_id() << " onNext " << v->GetFrame() << std::endl;
        },
        [](){
          std::cout << std::this_thread::get_id() << " OnCompleted" << std::endl;
        }
      );


    std::cout << std::this_thread::get_id() << " sleep" << std::endl;
    std::this_thread::sleep_for(5s);

    // Remove actors from the simulation.
    camera->Destroy();
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