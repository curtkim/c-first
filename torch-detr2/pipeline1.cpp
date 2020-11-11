#include <iostream>
#include <random>
#include <string>
#include <thread>

#include <readerwriterqueue.h>

#include "carla_common.hpp"
#include "viz_opengl.hpp"
#include "detr.hpp"

#include <carla/client/Sensor.h>
#include <carla/sensor/data/Image.h>


namespace cc = carla::client;
namespace cg = carla::geom;
namespace cs = carla::sensor;
namespace csd = carla::sensor::data;

static const std::string MAP_NAME = "/Game/Carla/Maps/Town03";



int main(int argc, const char *argv[]) {
  using namespace moodycamel;
  using namespace std::chrono;

  unsigned int WIDTH = 1024;
  unsigned int HEIGHT = 800;

  std::cout << "main thread: " << std::this_thread::get_id() << std::endl;

  auto torch_device = torch::Device(torch::kCUDA, 1);

  torch::jit::script::Module detr_model = detr::load_model("../../wrapped_detr_resnet50.pt", torch_device);



  ReaderWriterQueue<boost::shared_ptr<csd::Image>> q(2);

  auto[world, vehicle] = init_carla(MAP_NAME);
  auto blueprint_library = world.GetBlueprintLibrary();

  auto camera_transform = cg::Transform{
    cg::Location{1.5f, 0.0f, 2.8f},   // x, y, z.
    cg::Rotation{-1.0f, 0.0f, 0.0f}}; // pitch, yaw, roll.

  auto *camera_bp = blueprint_library->Find("sensor.camera.rgb");
  assert(camera_bp != nullptr);
  const_cast<cc::ActorBlueprint *>(camera_bp)->SetAttribute("sensor_tick", "0.033");
  const_cast<cc::ActorBlueprint *>(camera_bp)->SetAttribute("image_size_x", std::to_string(WIDTH));
  const_cast<cc::ActorBlueprint *>(camera_bp)->SetAttribute("image_size_y", std::to_string(HEIGHT));


  auto cam_actor = world.SpawnActor(*camera_bp, camera_transform, vehicle.get());
  auto camera = boost::static_pointer_cast<cc::Sensor>(cam_actor);


  GLFWwindow *window = make_window(WIDTH, HEIGHT);
  auto[VAO, VBO, EBO] = viz::bg::load_model();
  glBindVertexArray(VAO);

  Shader ourShader(viz::bg::VERTEX_SHADER_SOURCE, viz::bg::FRAGMENT_SHADER_SOURCE);
  ourShader.use();

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
    auto start_time = duration_cast<milliseconds>(system_clock::now().time_since_epoch());

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
    auto bounding_boxes = detr::detect(detr_model, img);
    std::cout << bounding_boxes << std::endl;

    auto end_time = duration_cast<milliseconds>(system_clock::now().time_since_epoch());
    long diff_detr = (end_time - start_time).count();


    // viz
    // ------
    glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    unsigned int texture = viz::bg::load_texture(pImage->GetWidth(), pImage->GetHeight(), pImage->data());
    glBindTexture(GL_TEXTURE_2D, texture);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
    glDeleteTextures(1, &texture);



    glfwSwapBuffers(window);
    glfwPollEvents();

    end_time = duration_cast<milliseconds>(system_clock::now().time_since_epoch());

    long diff = (end_time - start_time).count();
    std::cout << std::this_thread::get_id() << " " << diff << "ms " << diff_detr << "ms frame=" << pImage->GetFrame() << std::endl;
  }

  viz::bg::delete_model(VAO, VBO, EBO);

  // glfw: terminate, clearing all previously allocated GLFW resources.
  // ------------------------------------------------------------------
  glfwTerminate();

  camera->Stop();
  std::this_thread::sleep_for(milliseconds(1000));

  // Remove actors from the simulation.
  camera->Destroy();
  vehicle->Destroy();
  std::cout << std::this_thread::get_id() << " Actors destroyed." << std::endl;
}