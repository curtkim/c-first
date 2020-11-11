#include <iostream>
#include <random>
#include <string>
#include <thread>
#include <tuple>

#include <rxcpp/rx.hpp>

#include "common.hpp"
#include "carla_common.hpp"
#include "carla_rxcpp.hpp"
#include <carla/sensor/data/LidarMeasurement.h>
#include <carla/pointcloud/PointCloudIO.h>

#include "53_lidar_opengl.hpp"


namespace cc = carla::client;
namespace cg = carla::geom;
namespace cs = carla::sensor;
namespace csd = carla::sensor::data;

static const std::string MAP_NAME = "/Game/Carla/Maps/Town03";

auto load_model(boost::shared_ptr<csd::LidarMeasurement> lidar_measure) {
  GLuint VAO;
  glGenVertexArrays(1, &VAO);

  GLuint VBO;
  glGenBuffers(1, &VBO);
  glBindVertexArray(VAO);

  // Vertex
  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(float) * lidar_measure->size(), lidar_measure->data(), GL_STATIC_DRAW);

  glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(GLfloat), (GLvoid *)0);
  glEnableVertexAttribArray(0);
  glBindBuffer(GL_ARRAY_BUFFER, 0);
  glBindVertexArray(0);

  return std::make_tuple(VAO, VBO, lidar_measure->size());
}

constexpr int POINTS_PER_SECOND = 360*10*32;

int main(int argc, const char *argv[]) {

  std::cout << "main thread: " << std::this_thread::get_id() << std::endl;

  auto[world, vehicle] = init_carla(MAP_NAME);
  auto lidar_transform = cg::Transform{
    cg::Location{0.0f, 0.0f, 2.8f},     // x, y, z.
    cg::Rotation{0.0f, 0.0f, 0.0f}};    // pitch, yaw, roll.
  std::map<std::string, std::string> lidar_attributes = {
     {"sensor_tick", "0.1"},
     {"range", "200"},
     {"channels", "32"},
     {"rotation_frequency", "10"},
     {"points_per_second", std::to_string(POINTS_PER_SECOND)},
  };
  auto [lidar, lidar$] = from_sensor_data<csd::LidarMeasurement>(
    world, "sensor.lidar.ray_cast", lidar_attributes, lidar_transform,
    vehicle);

  // Apply control to vehicle.
  cc::Vehicle::Control control;
  control.throttle = 1.0f;
  vehicle->ApplyControl(control);

  rxcpp::schedulers::run_loop rl;

  rxcpp::subjects::subject<int> framebus;
  auto frame$ = framebus.get_observable();
  auto frameout = framebus.get_subscriber();
  auto sendFrame = [frameout](int frame) {
      frameout.on_next(frame);
  };

  GLFWwindow *window = make_window();

  // Move spectator so we can see the vehicle from the simulator window.
  auto spectator = world.GetSpectator();

  frame$
    .with_latest_from(lidar$)
    .tap([&spectator](std::tuple<int, boost::shared_ptr<csd::LidarMeasurement>> v) {

//      auto tf = vehicle->GetTransform();
//      tf.location.z += 2.0f;
//      spectator->SetTransform(tf);

      auto[frame, lidar_measure] = v;
      auto [VAO, VBO, point_length] = load_model(lidar_measure);

      glBindVertexArray(VAO);
      glDrawArrays(GL_POINTS, 0, point_length);

      glBindVertexArray(0);
      glDeleteVertexArrays(1, &VAO);
      glDeleteBuffers(1, &VBO);

      std::cout << "render: " << std::this_thread::get_id()
        << " time=" << getEpochMillisecond()
        << " frame=" << lidar_measure->GetFrame()
        << " VAO=" << VAO
        << std::endl;

      /*
      char buffer[9u];
      std::snprintf(buffer, sizeof(buffer), "%08zu", lidar_measure->GetFrame());
      auto filename = "_lidar/"s + buffer + ".ply";
      carla::pointcloud::PointCloudIO::SaveToDisk(filename, lidar_measure->begin(), lidar_measure->end());
      */

    })
    .subscribe();

  loop_opengl(window, rl, sendFrame);

  // Remove actors from the simulation.
  lidar->Destroy();
  vehicle->Destroy();
  std::cout << std::this_thread::get_id() << " Actors destroyed." << std::endl;
}