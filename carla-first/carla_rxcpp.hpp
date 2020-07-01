#pragma once

#include <chrono>

#include <rxcpp/rx-sources.hpp>

#include <carla/client/ActorBlueprint.h>
#include <carla/client/BlueprintLibrary.h>
#include <carla/client/Client.h>
#include <carla/client/Map.h>
#include <carla/client/Sensor.h>
#include <carla/client/TimeoutException.h>
#include <carla/geom/Transform.h>
#include <carla/image/ImageIO.h>
#include <carla/sensor/data/Image.h>
#include <carla/sensor/data/LidarMeasurement.h>
#include <carla/client/World.h>


namespace cc = carla::client;
namespace cs = carla::sensor;
namespace cg = carla::geom;

using namespace std::string_literals;
using namespace std::chrono_literals;

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

