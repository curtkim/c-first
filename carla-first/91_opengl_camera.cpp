#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <tuple>

#include <rxcpp/rx.hpp>

#include "common.hpp"
#include "carla_common.hpp"
#include "91_opengl.hpp"


namespace cc = carla::client;
namespace cg = carla::geom;
namespace cs = carla::sensor;
namespace csd = carla::sensor::data;

static const std::string MAP_NAME = "/Game/Carla/Maps/Town03";

unsigned int loadTexture(boost::shared_ptr<csd::Image> image) {
  unsigned int texture1;
  glGenTextures(1, &texture1);
  glBindTexture(GL_TEXTURE_2D, texture1);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image->GetWidth(), image->GetHeight(), 0, GL_BGRA, GL_UNSIGNED_BYTE, image->data());
  glGenerateMipmap(GL_TEXTURE_2D);
  return texture1;
}

int main(int argc, const char *argv[]) {

  std::cout << "main thread: " << std::this_thread::get_id() << std::endl;

  auto[world, vehicle] = init_carla(MAP_NAME);
  auto camera_transform = cg::Transform{
    cg::Location{-5.5f, 0.0f, 2.8f},   // x, y, z.
    cg::Rotation{-15.0f, 0.0f, 0.0f}}; // pitch, yaw, roll.
  std::map<std::string, std::string> caemra_attributes = {{"sensor_tick", "0.033"}};
  auto[camera, _image$] = from_sensor_data<csd::Image>(
    world, "sensor.camera.rgb", caemra_attributes, camera_transform,
    vehicle);

  // Apply control to vehicle.
  cc::Vehicle::Control control;
  control.throttle = 1.0f;
  vehicle->ApplyControl(control);


  auto image$ = _image$.tap([](boost::shared_ptr<csd::Image> image) {
      std::cout << "image: " << std::this_thread::get_id()
        << " time=" << getEpochMillisecond()
        << " frame=" << image->GetFrame()
        << " in tap" << std::endl;
  }).subscribe_on(rxcpp::observe_on_event_loop());


  rxcpp::schedulers::run_loop rl;

  rxcpp::subjects::subject<int> framebus;
  auto frame$ = framebus.get_observable();
  auto frameout = framebus.get_subscriber();
  auto sendFrame = [frameout](int frame) {
      frameout.on_next(frame);
  };

  GLFWwindow *window = make_window();
  auto[VAO, VBO, EBO] = load_model();
  glBindVertexArray(VAO);

  frame$
    .with_latest_from(image$)
    .tap([](std::tuple<int, boost::shared_ptr<csd::Image>> v) {
        auto[frame, image] = v;
        unsigned int texture = loadTexture(image);
        glBindTexture(GL_TEXTURE_2D, texture);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        glDeleteTextures(1, &texture);
        std::cout << "render: " << std::this_thread::get_id()
          << " time=" << getEpochMillisecond()
          << " frame=" << image->GetFrame()
          << " texture=" << texture << std::endl;
    })
    .subscribe();

  loop_opengl(window, rl, sendFrame);

  glDeleteVertexArrays(1, &VAO);
  glDeleteBuffers(1, &VBO);
  glDeleteBuffers(1, &EBO);

  // Remove actors from the simulation.
  camera->Destroy();
  vehicle->Destroy();
  std::cout << std::this_thread::get_id() << " Actors destroyed." << std::endl;
}