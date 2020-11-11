#include <iostream>
#include <random>
#include <string>
#include <thread>

#include <readerwriterqueue.h>

#include "carla_common.hpp"
#include "pipeline_opengl.hpp"
#include "detr.hpp"

#include <carla/client/Sensor.h>
#include <carla/sensor/data/Image.h>
#include <carla/sensor/data/LidarMeasurement.h>



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

  unsigned int WIDTH = 1024;
  unsigned int HEIGHT = 800;

  std::cout << "main thread: " << std::this_thread::get_id() << std::endl;

  auto torch_device = torch::Device(torch::kCUDA, 1);

  torch::jit::script::Module detr_module = detr::load_module("../../wrapped_detr_resnet50.pt", torch_device);



  ReaderWriterQueue<boost::shared_ptr<csd::Image>> q(2);

  auto[world, vehicle] = init_carla(MAP_NAME);
  auto blueprint_library = world.GetBlueprintLibrary();

  auto camera_transform = cg::Transform{
    cg::Location{1.5f, 0.0f, 2.8f},   // x, y, z.
    cg::Rotation{-1.0f, 0.0f, 0.0f}}; // pitch, yaw, roll.

  auto *camera_bp = blueprint_library->Find("sensor.camera.rgb");
  assert(camera_bp != nullptr);
  const_cast<carla::client::ActorBlueprint *>(camera_bp)->SetAttribute("sensor_tick", "0.033");
  const_cast<carla::client::ActorBlueprint *>(camera_bp)->SetAttribute("image_size_x", std::to_string(WIDTH));
  const_cast<carla::client::ActorBlueprint *>(camera_bp)->SetAttribute("image_size_y", std::to_string(HEIGHT));


  auto cam_actor = world.SpawnActor(*camera_bp, camera_transform, vehicle.get());
  auto camera = boost::static_pointer_cast<cc::Sensor>(cam_actor);


  GLFWwindow *window = make_window(WIDTH, HEIGHT);
  auto[VAO, VBO, EBO] = load_model();
  glBindVertexArray(VAO);

  Shader ourShader(MyConstants::VERTEX_SHADER_SOURCE, MyConstants::FRAGMENT_SHADER_SOURCE);
  ourShader.use();
  //ourShader.setInt("texture1", GL_TEXTURE0);

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


  int frame = 0;
  boost::shared_ptr<csd::Image> pImage;

  while (!glfwWindowShouldClose(window))
  {
    auto start_time = std::chrono::duration_cast< std::chrono::milliseconds >(std::chrono::system_clock::now().time_since_epoch());

    // input
    // -----
    processInput(window);

    // get from queue
    while(!q.try_dequeue(pImage)){}


    auto img = torch::from_blob(pImage->data(), {HEIGHT, WIDTH, 4}, torch::kUInt8)
      .clone()
      .to(torch::kFloat32)
      .permute({2, 0, 1})
      .index({torch::indexing::Slice(0, 3), torch::indexing::Ellipsis})
      .div_(255)
      .to(torch_device);
    //std::cout << img.sizes() << std::endl;
    auto bounding_boxes = detr::detect(detr_module, img);
    std::cout << bounding_boxes << std::endl;

    auto end_time = std::chrono::duration_cast< std::chrono::milliseconds >(std::chrono::system_clock::now().time_since_epoch());
    long diff_detr = (end_time - start_time).count();


    // render
    // ------
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    unsigned int texture = loadTexture(pImage);
    glBindTexture(GL_TEXTURE_2D, texture);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    glDeleteTextures(1, &texture);


    glfwSwapBuffers(window);
    glfwPollEvents();

    end_time = std::chrono::duration_cast< std::chrono::milliseconds >(std::chrono::system_clock::now().time_since_epoch());

    long diff = (end_time - start_time).count();
    std::cout << std::this_thread::get_id() << " " << diff << "ms " << diff_detr << "ms frame=" << pImage->GetFrame() << std::endl;
  }

  // glfw: terminate, clearing all previously allocated GLFW resources.
  // ------------------------------------------------------------------
  glfwTerminate();


  glDeleteVertexArrays(1, &VAO);
  glDeleteBuffers(1, &VBO);
  glDeleteBuffers(1, &EBO);

  camera->Stop();
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));

  // Remove actors from the simulation.
  camera->Destroy();
  vehicle->Destroy();
  std::cout << std::this_thread::get_id() << " Actors destroyed." << std::endl;
}