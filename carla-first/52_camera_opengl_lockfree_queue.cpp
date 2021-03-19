#include <iostream>
#include <random>
#include <string>
#include <thread>

#include "common.hpp"
#include "carla_common.hpp"
#include "50_camera_opengl.hpp"
#include "50_camera_opengl_carla.hpp" // loadTexture

#include <readerwriterqueue.h>

using namespace moodycamel;

namespace cc = carla::client;
namespace cg = carla::geom;
namespace cs = carla::sensor;
namespace csd = carla::sensor::data;


static const std::string MAP_NAME = "/Game/Carla/Maps/Town03";


int main(int argc, const char *argv[]) {

  std::cout << "main thread: " << std::this_thread::get_id() << std::endl;

  ReaderWriterQueue<boost::shared_ptr<cs::SensorData>> q(1);

  auto[world, vehicle] = init_carla(MAP_NAME);
  auto blueprint_library = world.GetBlueprintLibrary();

  auto camera_transform = cg::Transform{
    cg::Location{-5.5f, 0.0f, 2.8f},   // x, y, z.
    cg::Rotation{-15.0f, 0.0f, 0.0f}}; // pitch, yaw, roll.

  auto *camera_bp = blueprint_library->Find("sensor.camera.rgb");
  assert(camera_bp != nullptr);
  const_cast<carla::client::ActorBlueprint *>(camera_bp)->SetAttribute("sensor_tick", "0.033");

  auto cam_actor = world.SpawnActor(*camera_bp, camera_transform, vehicle.get());
  auto camera = boost::static_pointer_cast<cc::Sensor>(cam_actor);

  // Register a callback to save images to disk.
  camera->Listen([&q](auto data) {
    //auto image = boost::static_pointer_cast<csd::Image>(data);
    //assert(image != nullptr);
    //bool success = q.try_enqueue(image);
    bool success = q.try_enqueue(data);
    if( !success){
      // q max_size 2라서 loop가 꺼내가지 않으면 실패가 발생한다.
      std::cout << std::this_thread::get_id() << " fail enqueue frame=" << data->GetFrame() << std::endl;
    }
  });

  vehicle->SetAutopilot(true);


  GLFWwindow *window = make_window();
  auto[VAO, VBO, EBO] = load_model();
  glBindVertexArray(VAO);

  Shader ourShader(VERTEX_SHADER_SOURCE, FRAGMENT_SHADER_SOURCE);
  ourShader.use();
  //ourShader.setInt("texture1", GL_TEXTURE0);

  int frame = 0;
  boost::shared_ptr<cs::SensorData> pSensorData;

  while (!glfwWindowShouldClose(window))
  {
    auto start_time = std::chrono::duration_cast< std::chrono::milliseconds >(std::chrono::system_clock::now().time_since_epoch());

    // input
    // -----
    processInput(window);

    // render
    // ------
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    while(!q.try_dequeue(pSensorData)){}

    auto pImage = boost::static_pointer_cast<csd::Image>(pSensorData);
    unsigned int texture = loadTexture(pImage);
    glBindTexture(GL_TEXTURE_2D, texture);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    glDeleteTextures(1, &texture);


    glfwSwapBuffers(window);
    glfwPollEvents();

    auto end_time = std::chrono::duration_cast< std::chrono::milliseconds >(std::chrono::system_clock::now().time_since_epoch());

    long diff = (end_time - start_time).count();
    long duration = 1000/30;
    std::cout << std::this_thread::get_id() << " " << diff << "ms frame=" << pImage->GetFrame() << std::endl;
  }

  // glfw: terminate, clearing all previously allocated GLFW resources.
  // ------------------------------------------------------------------
  glfwTerminate();


  glDeleteVertexArrays(1, &VAO);
  glDeleteBuffers(1, &VBO);
  glDeleteBuffers(1, &EBO);

  // Remove actors from the simulation.
  camera->Destroy();
  vehicle->Destroy();
  std::cout << std::this_thread::get_id() << " Actors destroyed." << std::endl;
}