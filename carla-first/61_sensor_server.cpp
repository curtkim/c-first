#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <tuple>
#include <fstream>
#include <memory>

#include <asio.hpp>
#include <rxcpp/rx.hpp>

#include "common.hpp"
#include "carla_common.hpp"

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
  tcp::acceptor acceptor(io_context, tcp::endpoint(tcp::v4(), 8000));
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

    std::cout << "begin=" << image->begin() << " end=" << image->end()<< std::endl;
    std::cout << std::this_thread::get_id() << " " << getEpochMillisecond() << " frame=" << image->GetFrame() << " " << image->size() << std::endl;
    if( _socket )
      _socket->async_write_some(asio::buffer(image->data(), image->size()*4), [image](std::error_code ec, std::size_t length) {
        if (!ec) {
          std::cout << std::this_thread::get_id() << " " << getEpochMillisecond() << " frame=" << image->GetFrame() << " write" << std::endl;
        }
      });
  });

  // Apply control to vehicle.
  cc::Vehicle::Control control;
  control.throttle = 1.0f;
  vehicle->ApplyControl(control);


  /*
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
  */

  io_context.run();

  // Remove actors from the simulation.
  camera->Destroy();
  vehicle->Destroy();
  std::cout << std::this_thread::get_id() << " Actors destroyed." << std::endl;
}