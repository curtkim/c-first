#include <iostream>
#include <random>
#include <string>
#include <thread>

#include "common.hpp"
#include "carla_common.hpp"
#include "51_camera_opengl.hpp"
#include <readerwriterqueue.h>

using namespace moodycamel;

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

  ReaderWriterQueue<boost::shared_ptr<csd::Image>> q(2);

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
    auto image = boost::static_pointer_cast<csd::Image>(data);
    assert(image != nullptr);
    bool success = q.try_enqueue(image);
    if( !success){
      std::cout << std::this_thread::get_id() << " fail enqueue frame=" << image->GetFrame() << std::endl;
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
  boost::shared_ptr<csd::Image> pImage;

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

    while(!q.try_dequeue(pImage)){}

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