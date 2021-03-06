#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <fstream>
#include <memory>
#include <chrono>

#include <type_traits>

#include <asio.hpp>

#include "common.hpp"
#include "carla_common.hpp"
#include "70_header.hpp"
#include "70_sensors.hpp"

namespace cc = carla::client;
namespace cg = carla::geom;
namespace cs = carla::sensor;
namespace csd = carla::sensor::data;

static const std::string MAP_NAME = "/Game/Carla/Maps/Town03";

using asio::ip::tcp;

struct MyRecord {
  Header header;
  std::string topic_name;
  carla::SharedPtr<carla::sensor::SensorData> data;

  MyRecord(Header header, std::string topic_name, carla::SharedPtr<carla::sensor::SensorData> data) : header(header), topic_name(topic_name), data(data) {}
};


asio::io_context io_context;
std::deque<MyRecord> _write_msgs;
std::shared_ptr<tcp::socket> _socket;


void do_accept(tcp::acceptor &acceptor) {
  acceptor.async_accept([&acceptor](std::error_code ec, tcp::socket socket) {
    if (!ec) {
      std::cout << socket.local_endpoint() << std::endl;
      std::cout << socket.remote_endpoint() << std::endl;
      _socket = std::make_shared<tcp::socket>(std::move(socket));
    }
    do_accept(acceptor);
  });
}

auto make_listener(const std::string& topic_name) {
  return [&topic_name](carla::SharedPtr<carla::sensor::SensorData> data) {
    auto image = boost::static_pointer_cast<csd::Image>(data);
    assert(image != nullptr);

    std::cout << std::this_thread::get_id() << " camera " << getEpochMillisecond() << " frame=" << image->GetFrame() << " " << image->size() << std::endl;

    //myfile.write( (char*)image->data(), header.body_length );
    //myfile.flush();

    if( _socket ) {
      Header header;
      header.frame = image->GetFrame();
      header.body_length = image->size() * 4;
      header.topic_name_length = topic_name.length();

      std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
      std::chrono::duration<double> d = std::chrono::duration<double>(now.time_since_epoch());
      header.timepoint = d.count();
      header.record_type = 0;
      header.param1 = image->GetWidth();
      header.param2 = image->GetHeight();

      MyRecord rec {header, topic_name, data};

      /*
      std::vector<asio::const_buffer> buffers;
      buffers.push_back(asio::buffer(&header, sizeof(header)));
      buffers.push_back(asio::buffer(topic_name, header.topic_name_length));
      buffers.push_back(asio::buffer(image->data(), header.body_length));

      size_t frame = image->GetFrame();
      _socket->async_write_some(buffers, [frame](std::error_code ec, std::size_t length) {
        if (!ec) {
          std::cout << std::this_thread::get_id() << " camera " << getEpochMillisecond() << " frame=" << frame
                    << " write" << std::endl;
        }
      });
       */
    }
  };
}

int main(int argc, const char *argv[]) {

  tcp::acceptor acceptor(io_context, tcp::endpoint(tcp::v4(), 7000));
  do_accept(acceptor);

  asio::signal_set signals(io_context, SIGINT, SIGTERM);
  signals.async_wait([&](auto, auto){
    io_context.stop();
  });



  std::cout << "main thread: " << std::this_thread::get_id() << std::endl;

  auto[world, vehicle] = init_carla(MAP_NAME);
  auto blueprint_library = world.GetBlueprintLibrary();

  auto *camera_bp = blueprint_library->Find("sensor.camera.rgb");
  assert(camera_bp != nullptr);
  const_cast<carla::client::ActorBlueprint *>(camera_bp)->SetAttribute("sensor_tick", "0.033");

  std::vector<boost::shared_ptr<cc::Sensor>> cameras;
  for( auto const& [topic_name, transform] : CAMERA_TOPIC_TRANSFORM_MAP ) {
    auto cam_actor = world.SpawnActor(*camera_bp, transform, vehicle.get());
    auto camera = boost::static_pointer_cast<cc::Sensor>(cam_actor);
    cameras.push_back(camera);
    camera->Listen(make_listener(topic_name));
  }

  auto *lidar_bp = blueprint_library->Find("sensor.lidar.ray_cast");
  assert(lidar_bp != nullptr);

  constexpr int POINTS_PER_SECOND = 360*10*32;
  std::map<std::string, std::string> lidar_attributes = {
    {"sensor_tick", "0.1"},
    {"range", "200"},
    {"channels", "32"},
    {"rotation_frequency", "10"},
    {"points_per_second", std::to_string(POINTS_PER_SECOND)},
  };
  for( auto const& [key, val] : lidar_attributes )
    const_cast<carla::client::ActorBlueprint *>(lidar_bp)->SetAttribute(key, val);


  auto lidar_transform = cg::Transform{
    cg::Location{0.0f, 0.0f, 2.8f},   // x, y, z.
    cg::Rotation{0.0f, 0.0f, 0.0f}}; // pitch, yaw, roll.
  auto lidar_actor = world.SpawnActor(*lidar_bp, lidar_transform, vehicle.get());
  auto lidar = boost::static_pointer_cast<cc::Sensor>(lidar_actor);

  lidar->Listen([](auto data) {
    auto lidar_data = boost::static_pointer_cast<csd::LidarMeasurement>(data);
    assert(lidar_data != nullptr);

    std::cout << std::this_thread::get_id() << " lidar " << getEpochMillisecond() << " frame=" << lidar_data->GetFrame() << " " << lidar_data->size() << std::endl;

    std::string topic_name = "/lidar/0";

    //myfile.write( (char*)lidar_data->data(), header.body_length );
    //myfile.flush();

    if( _socket ) {
      Header header;
      header.frame = lidar_data->GetFrame();
      header.body_length = lidar_data->size() * 3 * sizeof(float);
      header.topic_name_length = topic_name.length();


      std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
      std::chrono::duration<double> d = std::chrono::duration<double>(now.time_since_epoch());
      header.timepoint = d.count();
      header.record_type = 1;

      MyRecord rec {header, topic_name, data};
      /*
      std::vector<asio::const_buffer> buffers;
      buffers.push_back(asio::buffer(&header, sizeof(header)));
      buffers.push_back(asio::buffer(topic_name, header.topic_name_length));
      buffers.push_back(asio::buffer(lidar_data->data(), header.body_length));

      size_t frame = lidar_data->GetFrame();
      _socket->async_write_some(buffers, [frame](std::error_code ec, std::size_t length) {
        if (!ec) {
          std::cout << std::this_thread::get_id() << " lidar " << getEpochMillisecond() << " frame=" << frame
                    << " write" << std::endl;
        }
      });
       */
    }
  });

  // autopilot
  vehicle->SetAutopilot(true);

  io_context.run();

  for(std::vector<boost::shared_ptr<cc::Sensor>>::iterator it = cameras.begin(); it != cameras.end(); ++it) {
    (*it)->Stop();
    (*it)->Destroy();
  }

  lidar->Stop();
  lidar->Destroy();
  vehicle->Destroy();

  std::cout << std::this_thread::get_id() << " Actors destroyed." << std::endl;
}