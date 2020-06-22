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
#include "70_header.hpp"

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

int main(int argc, const char *argv[]) {

  // asio
  asio::io_context io_context;
  tcp::acceptor acceptor(io_context, tcp::endpoint(tcp::v4(), 7000));
  do_accept(acceptor);

  std::cout << "main thread: " << std::this_thread::get_id() << std::endl;

  auto[world, vehicle] = init_carla(MAP_NAME);
  auto blueprint_library = world.GetBlueprintLibrary();

  auto *camera_bp = blueprint_library->Find("sensor.camera.rgb");
  assert(camera_bp != nullptr);
  const_cast<carla::client::ActorBlueprint *>(camera_bp)->SetAttribute("sensor_tick", "0.033");

  // Spawn a camera attached to the vehicle.
  auto camera_transform = cg::Transform{
    cg::Location{-5.5f, 0.0f, 2.8f},   // x, y, z.
    cg::Rotation{-15.0f, 0.0f, 0.0f}}; // pitch, yaw, roll.
  auto cam_actor = world.SpawnActor(*camera_bp, camera_transform, vehicle.get());
  auto camera = boost::static_pointer_cast<cc::Sensor>(cam_actor);

  // Register a callback to save images to disk.
  camera->Listen([](auto data) {
    auto image = boost::static_pointer_cast<csd::Image>(data);
    assert(image != nullptr);

    std::cout << std::this_thread::get_id() << " " << getEpochMillisecond() << " frame=" << image->GetFrame() << " " << image->size() << std::endl;
    if( _socket ) {
      std::string topic_name = "/camera/0";
      Header header;
      header.frame = image->GetFrame();
      header.body_length = image->size() * 4;
      header.topic_name_length = topic_name.length();

      std::chrono::system_clock::time_point now = std::chrono::system_clock::now();
      std::chrono::duration<double> d = std::chrono::duration<double>(now.time_since_epoch());
      header.timepoint = d.count();
      header.record_type = 0;

      std::vector<asio::const_buffer> buffers;
      buffers.push_back(asio::buffer(&header, sizeof(header)));
      buffers.push_back(asio::buffer(topic_name, header.topic_name_length));
      buffers.push_back(asio::buffer(image->data(), header.body_length));

      size_t frame = image->GetFrame();
      _socket->async_write_some(buffers, [frame](std::error_code ec, std::size_t length) {
        if (!ec) {
          std::cout << std::this_thread::get_id() << " " << getEpochMillisecond() << " frame=" << frame
                    << " write" << std::endl;
        }
      });
    }
  });

  vehicle->SetAutopilot(true);

  io_context.run();

  // Remove actors from the simulation.
  camera->Destroy();
  vehicle->Destroy();
  std::cout << std::this_thread::get_id() << " Actors destroyed." << std::endl;
}