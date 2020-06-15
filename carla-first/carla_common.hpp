#pragma once

#include <chrono>

namespace cc = carla::client;
namespace cs = carla::sensor;
namespace cg = carla::geom;

using namespace std::string_literals;
using namespace std::chrono_literals;

auto init_carla(std::string map_name) {
  auto client = cc::Client("localhost", 2000, 1);
  client.SetTimeout(10s);

  std::cout << "Client API version : " << client.GetClientVersion() << '\n';
  std::cout << "Server API version : " << client.GetServerVersion() << '\n';

  auto world = client.GetWorld();
  if (!ends_with(map_name, world.GetMap()->GetName())) {
    std::cout << "load map " << map_name << std::endl;
    world = client.LoadWorld(map_name);
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
  return std::make_tuple(world, vehicle);
}


/*
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
      });//.subscribe_on(rxcpp::synchronize_new_thread());
  return data$;
}
*/

template <class DataType>
auto from_sensor2(boost::shared_ptr<cc::Sensor> pSensor) {
  auto data$ = rxcpp::sources::create<boost::shared_ptr<DataType>>(
      [pSensor](rxcpp::subscriber<boost::shared_ptr<DataType>> s){
        std::cout << std::this_thread::get_id() << " before listen " << std::endl;
        pSensor->Listen([s](auto data){
          assert(data != nullptr);
          s.on_next(boost::static_pointer_cast<DataType>(data));
        });
      });
  return data$;
}

template <class DataType>
auto from_sensor_data(cc::World world, std::string sensor_type, std::map<std::string, std::string> attributes, cg::Transform tf, boost::shared_ptr<cc::Vehicle> vehicle) {
  auto blueprint_library = world.GetBlueprintLibrary();
  const cc::ActorBlueprint *sensor_bp = blueprint_library->Find(sensor_type);
  assert(sensor_bp != nullptr);
  for( auto const& [key, val] : attributes )
    const_cast<cc::ActorBlueprint *>(sensor_bp)->SetAttribute(key, val);

  auto sensor_actor = world.SpawnActor(*sensor_bp, tf, vehicle.get());
  auto sensor = boost::static_pointer_cast<cc::Sensor>(sensor_actor);

  auto data$ = from_sensor2<DataType>(sensor);
  return std::make_tuple(sensor, data$);
}

