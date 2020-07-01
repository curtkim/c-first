#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <tuple>
#include <fstream>
#include <memory>
#include <chrono>

#include <type_traits>

#include <asio.hpp>
#include <rxcpp/rx.hpp>

#include "common.hpp"
#include "carla_common.hpp"
#include "carla_rxcpp.hpp"
#include "70_header.hpp"
#include "70_sensors.hpp"

namespace cc = carla::client;
namespace cg = carla::geom;
namespace cs = carla::sensor;
namespace csd = carla::sensor::data;

static const std::string MAP_NAME = "/Game/Carla/Maps/Town03";

using asio::ip::tcp;

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

auto make_camera_subscriber(const std::string& topic_name, std::ofstream& myfile) {
  return [&topic_name, &myfile](boost::shared_ptr<csd::Image> image){
    std::cout << std::this_thread::get_id() << " " << topic_name << " " << getEpochMillisecond() << " frame=" << image->GetFrame() << " " << image->size() << std::endl;

    //myfile.write( (char*)image->data(), header.body_length );
    //myfile.flush();

    if( _socket ) {
      size_t frame = image->GetFrame();

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

      std::vector<asio::const_buffer> buffers;
      buffers.push_back(asio::buffer(&header, sizeof(header)));
      buffers.push_back(asio::buffer(topic_name, header.topic_name_length));
      buffers.push_back(asio::buffer(image->data(), header.body_length));

      std::size_t length = asio::write(*_socket.get(), buffers);
      std::cout << std::this_thread::get_id() << " " << topic_name << " " << getEpochMillisecond() << " frame=" << frame
                << " write len=" << length << " " << header.timepoint - image->GetTimestamp() << std::endl;
       /*
      strand.post([image, topic_name](){
        //std::cout << std::this_thread::get_id() << " " << topic_name << " " << getEpochMillisecond() << " frame=" << image->GetFrame() << " " << image->size() << std::endl;

        _socket->async_write_some(buffers, [frame, topic_name](std::error_code ec, std::size_t length) {
          if (!ec) {
            std::cout << std::this_thread::get_id() << " " << topic_name << " " << getEpochMillisecond() << " frame=" << frame
                      << " write len=" << length << std::endl;
          }
        });

      });
       */
    }
  };
}

auto make_lidar_subscriber(std::string&& topic_name, std::ofstream & myfile) {
  return [topic_name = std::move(topic_name), &myfile](boost::shared_ptr<csd::LidarMeasurement> lidar_data){
    std::cout << std::this_thread::get_id() << " " << topic_name << " " << getEpochMillisecond() << " frame=" << lidar_data->GetFrame() << " " << lidar_data->size() << std::endl;


    //myfile.write( (char*)lidar_data->data(), header.body_length );
    //myfile.flush();

    if( _socket ) {
      size_t frame = lidar_data->GetFrame();

      Header header;
      header.frame = lidar_data->GetFrame();
      header.body_length = lidar_data->size() * 3 * sizeof(float);
      header.topic_name_length = topic_name.length();

      std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
      std::chrono::duration<double> d = std::chrono::duration<double>(now.time_since_epoch());
      header.timepoint = d.count();
      header.record_type = 1;

      std::vector<asio::const_buffer> buffers;
      buffers.push_back(asio::buffer(&header, sizeof(header)));
      buffers.push_back(asio::buffer(topic_name, header.topic_name_length));
      buffers.push_back(asio::buffer(lidar_data->data(), header.body_length));

      std::size_t length = asio::write(*_socket.get(), buffers);
      std::cout << std::this_thread::get_id() << " " << topic_name << " " << getEpochMillisecond() << " frame=" << frame
                << " write len=" << length << " " << header.timepoint - lidar_data->GetTimestamp() << std::endl;
      /*
      strand.post([lidar_data, topic_name](){
        _socket->async_write_some(buffers, [topic_name, frame](std::error_code ec, std::size_t length) {
          if (!ec) {
            std::cout << std::this_thread::get_id() << " " << topic_name << " " << getEpochMillisecond() << " frame=" << frame
                      << " write len=" << length << std::endl;
          }
        });
      });
       */
    }
  };
}

int main(int argc, const char *argv[]) {

  // asio
  asio::io_context io_context;
  asio::io_context::strand strand(io_context);

  tcp::acceptor acceptor(io_context, tcp::endpoint(tcp::v4(), 7000));
  do_accept(acceptor);

  asio::signal_set signals(io_context, SIGINT, SIGTERM);
  signals.async_wait([&](auto, auto){
    io_context.stop();
  });

  std::ofstream myfile ("/data/dump.om", std::ios::out | std::ios::binary); // std::ios::app |


  std::cout << "main thread: " << std::this_thread::get_id() << std::endl;

  auto[world, vehicle] = init_carla(MAP_NAME);
  auto blueprint_library = world.GetBlueprintLibrary();

  std::vector<boost::shared_ptr<cc::Sensor>> cameras;
  std::map<std::string, std::string> camera_attributes = {{"sensor_tick", "0.033"}};

  for( auto const& [topic_name, transform] : CAMERA_TOPIC_TRANSFORM_MAP ){
    auto [camera, camera$] = from_sensor_data<csd::Image>(
      world, "sensor.camera.rgb", camera_attributes, transform, vehicle);
    camera$.subscribe(make_camera_subscriber(topic_name, myfile));
    cameras.push_back(camera);
  }


  auto lidar_transform = cg::Transform{
    cg::Location{-5.5f, 0.0f, 2.8f},   // x, y, z.
    cg::Rotation{-15.0f, 0.0f, 0.0f}}; // pitch, yaw, roll.
  constexpr int POINTS_PER_SECOND = 360*10*32;
  std::map<std::string, std::string> lidar_attributes = {
    {"sensor_tick", "0.1"},
    {"range", "200"},
    {"channels", "32"},
    {"rotation_frequency", "10"},
    {"points_per_second", std::to_string(POINTS_PER_SECOND)},
  };
  auto [lidar, lidar$] = from_sensor_data<csd::LidarMeasurement>(
    world, "sensor.lidar.ray_cast", lidar_attributes, lidar_transform, vehicle);

  lidar$.subscribe(make_lidar_subscriber("/lidar/0", myfile));


  vehicle->SetAutopilot(true);

  io_context.run();

  myfile.close();

  // Remove actors from the simulation.
  lidar->Stop();
  lidar->Destroy();

  for(std::vector<boost::shared_ptr<cc::Sensor>>::iterator it = cameras.begin(); it != cameras.end(); ++it) {
    (*it)->Stop();
    (*it)->Destroy();
  }

  vehicle->Destroy();

  std::cout << std::this_thread::get_id() << " Actors destroyed." << std::endl;
}